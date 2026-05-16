use egui_wgpu::{Renderer as EguiRenderer, RendererOptions, ScreenDescriptor};
use egui_winit::State as EguiWinitState;
use eluna::{
    build_d3d_triangle_strips, collect_emote_runtime_pipeline, collect_emote_timelines,
    collect_emote_variables, emote_ticks_to_milliseconds, expand_triangle_strips_to_list,
    milliseconds_to_emote_ticks, transform_order_mask, ElunaPlayer, EmoteDeviceRenderOptions,
    EmoteDrawPass, EmoteModelSchema, EmotePlayerControl, EmoteSceneBounds, EmoteStaticScene,
    EmoteStaticSprite, EmoteVertex, PhysicsControl, PsbDecryptionKey, PsbFile, PsbNormalizeOptions,
    TimelinePlayMode, VariableWrite, EMOTE_UPDATE_MS_CAP,
};
use image::GenericImageView;
use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// Single-shot per-draw dump for the first GPU draw stream of the process.
// Triggered on the first call to build_scene_draws so the dump matches the
// initial drawFrameInfo emission rather than a subsequent rebuild.  Use
// `ELUNA_DUMP_DRAWS=0` to suppress; left enabled by default because the
// brief explicitly asks for this diagnostic and one frame's worth of stderr
// is not noisy.
static GPU_DRAW_DUMP_DONE: AtomicBool = AtomicBool::new(false);
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, OwnedDisplayHandle};
use winit::keyboard::{Key, NamedKey};
use winit::window::Window;

static ELUNA_EMBEDDED_DEFAULT_FONT: &[u8] = include_bytes!("default.ttf");

const EGUI_DEFAULT_FONT_NAME: &str = "eluna_default_cjk";

const SHADER: &str = r#"
struct Transform {
    center: vec2<f32>,
    viewport_scale: vec2<f32>,
    player_coord: vec2<f32>,
    player_scale: f32,
    player_cos: f32,
    player_sin: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<uniform> transform: Transform;

@group(1) @binding(0)
var sprite_tex: texture_2d<f32>;

@group(1) @binding(1)
var sprite_sampler: sampler;

struct VertexIn {
    @location(0) position: vec2<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    var out: VertexOut;
    let p0 = (input.position - transform.center) * transform.player_scale;
    let p1 = vec2<f32>(
        p0.x * transform.player_cos - p0.y * transform.player_sin,
        p0.x * transform.player_sin + p0.y * transform.player_cos,
    ) + transform.player_coord;
    let p = p1 * transform.viewport_scale;
    out.position = vec4<f32>(p.x, -p.y, 0.0, 1.0);
    out.texcoord = input.texcoord;
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    var c = textureSample(sprite_tex, sprite_sampler, input.texcoord) * input.color;
    if (c.a <= 0.003) {
        discard;
    }
    return vec4<f32>(c.rgb * c.a, c.a);
}
"#;

const MASKED_SHADER: &str = r#"
struct Transform {
    center: vec2<f32>,
    viewport_scale: vec2<f32>,
    player_coord: vec2<f32>,
    player_scale: f32,
    player_cos: f32,
    player_sin: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<uniform> transform: Transform;

@group(1) @binding(0)
var sprite_tex: texture_2d<f32>;

@group(1) @binding(1)
var sprite_sampler: sampler;

@group(2) @binding(0)
var mask_tex: texture_2d<f32>;

@group(2) @binding(1)
var mask_sampler: sampler;

struct VertexIn {
    @location(0) position: vec2<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) mask_uv: vec2<f32>,
};

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    var out: VertexOut;
    let p0 = (input.position - transform.center) * transform.player_scale;
    let p1 = vec2<f32>(
        p0.x * transform.player_cos - p0.y * transform.player_sin,
        p0.x * transform.player_sin + p0.y * transform.player_cos,
    ) + transform.player_coord;
    let p = p1 * transform.viewport_scale;
    let ndc = vec2<f32>(p.x, -p.y);
    out.position = vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
    out.texcoord = input.texcoord;
    out.color = input.color;
    out.mask_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return out;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    var c = textureSample(sprite_tex, sprite_sampler, input.texcoord) * input.color;
    let m = textureSample(mask_tex, mask_sampler, input.mask_uv).a;
    let out_alpha = c.a * m;
    if (out_alpha <= 0.003) {
        discard;
    }
    return vec4<f32>(c.rgb * out_alpha, out_alpha);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSpriteVertex {
    position: [f32; 2],
    texcoord: [f32; 2],
    color: [f32; 4],
}

impl GpuSpriteVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] = [
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x2,
        },
        wgpu::VertexAttribute {
            offset: 8,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32x2,
        },
        wgpu::VertexAttribute {
            offset: 16,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32x4,
        },
    ];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuSpriteVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }

    fn from_emote_preview(vertex: EmoteVertex) -> Self {
        Self {
            position: [vertex.x, vertex.y],
            texcoord: [0.0, 0.0],
            color: vertex.diffuse_rgba_f32(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TransformUniform {
    center: [f32; 2],
    viewport_scale: [f32; 2],
    player_coord: [f32; 2],
    player_scale: f32,
    player_cos: f32,
    player_sin: f32,
    _pad: f32,
}

struct Options {
    input: Option<PathBuf>,
    normalize: PsbNormalizeOptions,
    motion: Option<String>,
    timeline: Option<String>,
    timeline_time: Option<f32>,
    no_timeline: bool,
    initial_writes: Vec<VariableWrite>,
    control_variable: Option<String>,
    dump_frames: Option<PathBuf>,
    gpu_dump: Option<PathBuf>,
    debug_frames: bool,
}

struct LoadedModel {
    normalized_data: Vec<u8>,
    psb: PsbFile,
    schema: EmoteModelSchema,
    active_motion: Option<String>,
    scene: EmoteStaticScene,
    player: ElunaPlayer,
    debug_frame_count: usize,
    debug_frames: bool,
}

struct UiState {
    show_motion: bool,
    show_variables: bool,
    show_selector: bool,
    show_face: bool,
    show_physics: bool,
    show_layers: bool,
    show_textures: bool,
    show_api_log: bool,
    var_filter: String,
    layer_filter: String,
    layer_filter_face_only: bool,
    layer_filter_visible_only: bool,
    selected_layer: Option<String>,
    playback_speed: f32,
    timeline_loop: bool,
    main_timeline: String,
    diff_timeline_slots: [String; 6],
    diff_fadeout_ms: f32,
    step_update: bool,
    transform_order_mask: u32,
    api_log_text: String,
    dirty: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            show_motion: true,
            show_variables: true,
            show_selector: false,
            show_face: false,
            show_physics: false,
            show_layers: false,
            show_textures: false,
            show_api_log: false,
            var_filter: String::new(),
            layer_filter: String::new(),
            layer_filter_face_only: false,
            layer_filter_visible_only: false,
            selected_layer: None,
            playback_speed: 1.0,
            timeline_loop: true,
            main_timeline: String::new(),
            diff_timeline_slots: Default::default(),
            diff_fadeout_ms: 300.0,
            step_update: false,
            transform_order_mask: transform_order_mask::DEFAULT,
            api_log_text: String::new(),
            dirty: false,
        }
    }
}

struct EguiOutput {
    paint_jobs: Vec<egui::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
    pixels_per_point: f32,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let options = parse_args()?;
    let mut loaded_model = if let Some(path) = &options.input {
        let input_data = fs::read(path)?;
        let (normalized_data, psb) = PsbFile::parse_normalized(&input_data, &options.normalize)?;
        let schema = EmoteModelSchema::from_psb(&psb)?;
        let active_motion = options
            .motion
            .clone()
            .or_else(|| schema.default_motion_name(&psb).ok().flatten());
        let variables = collect_emote_variables(&psb);
        let timelines = collect_emote_timelines(&psb);
        let runtime_pipeline = collect_emote_runtime_pipeline(&psb);
        let default_variable_values = default_variable_values(&variables, &timelines);
        let mut scene = if let Some(motion) = active_motion.as_deref() {
            schema.build_motion_scene_at_with_resources_and_variables(
                &psb,
                &normalized_data,
                motion,
                0.0,
                &default_variable_values,
            )?
        } else {
            schema.build_static_scene(&psb)?
        };
        let mut player = ElunaPlayer::from_scene_variables_timelines_runtime(
            scene.clone(),
            variables,
            timelines,
            runtime_pipeline,
        );
        for write in &options.initial_writes {
            player.apply_write(write.clone());
        }
        if !options.no_timeline {
            let timeline_to_play = options.timeline.clone().or_else(|| {
                active_motion
                    .as_deref()
                    .and_then(|motion| find_timeline_for_motion(&player, motion))
            });
            if let Some(name) = timeline_to_play {
                player.play_timeline(&name, TimelinePlayMode::PARALLEL.with_looping(true));
                if let Some(ticks) = options.timeline_time {
                    player
                        .set_timeline_time(&name, ticks)
                        .map_err(|err| IoError::new(ErrorKind::InvalidInput, err))?;
                }
                println!("active main timeline: {name}");
            }
        }
        if let Some(motion) = active_motion.as_deref() {
            scene = schema.build_motion_scene_at_with_resources_and_variables(
                &psb,
                &normalized_data,
                motion,
                active_main_timeline_time_ticks(&player).unwrap_or(0.0),
                &player_variable_values(&player),
            )?;
            player.replace_scene(scene.clone());
        }
        println!(
            "loaded Emote PSB: version={} resources={} motion={:?} sprites={} variables={} timelines={} controllers={}/{}/{}/{} drawFrameInfo={} bounds={:?}",
            psb.version,
            psb.resources.len(),
            active_motion,
            scene.sprites.iter().filter(|s| s.visible).count(),
            player.variables().len(),
            player.timelines().len(),
            player.runtime_pipeline().selector_controls.len(),
            player.runtime_pipeline().clamp_controls.len(),
            player.runtime_pipeline().physics_controls.len(),
            player.runtime_pipeline().transition_controls.len(),
            scene.draw_frame_info.len(),
            scene.bounds
        );
        if let Some(name) = &options.control_variable {
            println!("interactive variable: {name}  (A/D adjust, R reset, K print value)");
        }
        Some(LoadedModel {
            normalized_data,
            psb,
            schema,
            active_motion,
            scene,
            player,
            debug_frame_count: 0,
            debug_frames: options.debug_frames,
        })
    } else {
        None
    };

    if let (Some(model), Some(dir)) = (loaded_model.as_mut(), options.dump_frames.as_ref()) {
        dump_runtime_frames(model, dir, 30)?;
        return Ok(());
    }

    if let (Some(model), Some(dir)) = (loaded_model.as_ref(), options.gpu_dump.as_ref()) {
        dump_gpu_frame(model, dir)?;
        return Ok(());
    }

    let event_loop = EventLoop::new()?;
    let mut app = App {
        loaded_model,
        control_variable: options.control_variable.clone(),
        ..Default::default()
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

fn parse_args() -> Result<Options, Box<dyn Error>> {
    let mut input = None;
    let mut motion = None;
    let mut timeline = None;
    let mut timeline_time = None;
    let mut no_timeline = false;
    let mut key: Option<u32> = None;
    let mut initial_writes = Vec::new();
    let mut control_variable = None;
    let mut dump_frames = None;
    let mut gpu_dump = None;
    let mut debug_frames = false;
    let mut decode_mdf = true;
    let mut decode_lz4 = true;

    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let text = arg.to_string_lossy();
        match text.as_ref() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "--input" => input = Some(next_path_arg(&mut args, "--input")?),
            "--motion" => motion = Some(next_string_arg(&mut args, "--motion")?),
            "--timeline" => timeline = Some(next_string_arg(&mut args, "--timeline")?),
            "--timeline-time" => {
                timeline_time = Some(next_string_arg(&mut args, "--timeline-time")?.parse()?)
            }
            "--no-timeline" => no_timeline = true,
            "--set" => initial_writes.push(parse_variable_write(
                &next_string_arg(&mut args, "--set")?,
                false,
            )?),
            "--set-timed" => initial_writes.push(parse_variable_write(
                &next_string_arg(&mut args, "--set-timed")?,
                true,
            )?),
            "--control-variable" => {
                control_variable = Some(next_string_arg(&mut args, "--control-variable")?)
            }
            "--dump-frames" => dump_frames = Some(next_path_arg(&mut args, "--dump-frames")?),
            "--gpu-dump" => gpu_dump = Some(next_path_arg(&mut args, "--gpu-dump")?),
            "--debug-frames" => debug_frames = true,
            "--no-mdf" => decode_mdf = false,
            "--no-lz4" => decode_lz4 = false,
            "--key" => key = Some(next_u32_arg(&mut args, "--key")?),
            _ if text.starts_with("--input=") => input = Some(PathBuf::from(&text[8..])),
            _ if text.starts_with("--motion=") => motion = Some(text[9..].to_owned()),
            _ if text.starts_with("--timeline=") => timeline = Some(text[11..].to_owned()),
            _ if text.starts_with("--timeline-time=") => {
                timeline_time = Some(text[16..].parse()?)
            }
            _ if text.starts_with("--set=") => {
                initial_writes.push(parse_variable_write(&text[6..], false)?)
            }
            _ if text.starts_with("--set-timed=") => {
                initial_writes.push(parse_variable_write(&text[12..], true)?)
            }
            _ if text.starts_with("--control-variable=") => {
                control_variable = Some(text[19..].to_owned())
            }
            _ if text.starts_with("--dump-frames=") => {
                dump_frames = Some(PathBuf::from(&text[14..]))
            }
            _ if text.starts_with("--gpu-dump=") => gpu_dump = Some(PathBuf::from(&text[11..])),
            _ if text.starts_with("--key=") => key = Some(parse_u32_value(&text[6..])?),
            _ if !text.starts_with('-') && input.is_none() => {
                input = Some(PathBuf::from(text.as_ref()))
            }
            _ => {
                return Err(Box::new(IoError::new(
                    ErrorKind::InvalidInput,
                    "unexpected argument; use --input <model.psb>",
                )));
            }
        }
    }

    let decrypt_key = key.map(PsbDecryptionKey::emote_key);

    Ok(Options {
        input,
        normalize: PsbNormalizeOptions {
            decrypt_key,
            decode_mdf,
            decode_lz4,
        },
        motion,
        timeline,
        timeline_time,
        no_timeline,
        initial_writes,
        control_variable,
        dump_frames,
        gpu_dump,
        debug_frames,
    })
}

fn player_variable_values(player: &ElunaPlayer) -> BTreeMap<String, f32> {
    player
        .variables()
        .iter()
        .map(|(name, state)| (name.clone(), state.value))
        .collect()
}

fn rebuild_loaded_model_scene(
    model: &mut LoadedModel,
    physics_delta_ticks: f32,
) -> Result<(), Box<dyn Error>> {
    if let Some(motion_name) = model.active_motion.as_deref() {
        let variable_values = player_variable_values(&model.player);
        let scene_time_ticks = active_main_timeline_time_ticks(&model.player)
            .unwrap_or_else(|| model.player.elapsed_ticks());
        let scene = model
            .schema
            .build_motion_scene_at_with_resources_and_variables(
                &model.psb,
                &model.normalized_data,
                motion_name,
                scene_time_ticks,
                &variable_values,
            )?;
        model.scene = scene;
        model.player.replace_scene(model.scene.clone());
        if physics_delta_ticks > 0.0 && model.player.is_physics_enabled() {
            model
                .player
                .evaluate_physics_for_current_scene(physics_delta_ticks);
            let variable_values = player_variable_values(&model.player);
            let scene = model
                .schema
                .build_motion_scene_at_with_resources_and_variables(
                    &model.psb,
                    &model.normalized_data,
                    motion_name,
                    scene_time_ticks,
                    &variable_values,
                )?;
            model.scene = scene;
            model.player.replace_scene(model.scene.clone());
        }
    }
    Ok(())
}

fn active_main_timeline_time_ticks(player: &ElunaPlayer) -> Option<f32> {
    player
        .active_timelines()
        .iter()
        .find(|(name, mode)| !name.starts_with("@control/") && !mode.is_difference())
        .map(|(name, _)| player.timeline_elapsed_ticks(name))
}

fn default_variable_values(
    infos: &[eluna::EmoteVariableInfo],
    timelines: &[eluna::EmoteTimeline],
) -> BTreeMap<String, f32> {
    let mut values: BTreeMap<String, f32> = infos
        .iter()
        .map(|info| (info.name.clone(), info.default_value))
        .collect();
    for timeline in timelines {
        for variable in &timeline.variables {
            if let Some(first) = variable.frames.first() {
                values.insert(variable.name.clone(), first.value);
            }
        }
    }
    values
}

fn find_timeline_for_motion(player: &ElunaPlayer, motion: &str) -> Option<String> {
    if player.timelines().contains_key(motion) {
        return Some(motion.to_owned());
    }
    let suffix = format!("/{motion}");
    if let Some(name) = player
        .timelines()
        .values()
        .filter(|timeline| !timeline.name.starts_with("@control/") && !timeline.is_difference)
        .map(|timeline| &timeline.name)
        .find(|name| name.ends_with(&suffix))
        .cloned()
    {
        return Some(name);
    }
    player
        .timelines()
        .values()
        .filter(|timeline| !timeline.name.starts_with("@control/") && !timeline.is_difference)
        .map(|timeline| timeline.name.clone())
        .next()
}

fn parse_variable_write(text: &str, timed: bool) -> Result<VariableWrite, Box<dyn Error>> {
    let (name, rest) = text.split_once('=').ok_or_else(|| {
        IoError::new(
            ErrorKind::InvalidInput,
            "variable write must be name=value or name=value,time,easing",
        )
    })?;
    if name.is_empty() {
        return Err(Box::new(IoError::new(
            ErrorKind::InvalidInput,
            "variable name is empty",
        )));
    }
    let mut values = rest.split(',');
    let value: f32 = values
        .next()
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "missing variable value"))?
        .parse()?;
    if timed {
        let time_ticks: f32 = values.next().unwrap_or("0").parse()?;
        let easing: f32 = values.next().unwrap_or("0").parse()?;
        Ok(VariableWrite::timed(name, value, time_ticks, easing))
    } else {
        Ok(VariableWrite::immediate(name, value))
    }
}

fn next_string_arg(
    args: &mut impl Iterator<Item = OsString>,
    option: &'static str,
) -> Result<String, Box<dyn Error>> {
    let value = args
        .next()
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, option))?;
    Ok(value.to_string_lossy().into_owned())
}

fn next_path_arg(
    args: &mut impl Iterator<Item = OsString>,
    option: &'static str,
) -> Result<PathBuf, Box<dyn Error>> {
    let value = args
        .next()
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, option))?;
    Ok(PathBuf::from(value))
}

fn next_u32_arg(
    args: &mut impl Iterator<Item = OsString>,
    option: &'static str,
) -> Result<u32, Box<dyn Error>> {
    let value = args
        .next()
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, option))?;
    parse_u32_value(&value.to_string_lossy())
}

fn parse_u32_value(value: &str) -> Result<u32, Box<dyn Error>> {
    let trimmed = value.trim();
    let parsed = if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u32::from_str_radix(hex, 16)?
    } else {
        trimmed.parse::<u32>()?
    };
    Ok(parsed)
}

fn print_usage() {
    println!("usage: eluna_player [--input <model.psb|model.mdf|model.lz4>] [options]");
    println!();
    println!("preview options:");
    println!("  --motion <name>      select the active motion for dynamic frameList playback");
    println!("  --timeline <name>    explicitly play a metadata/timelineControl timeline");
    println!("  --timeline-time <t>  scrub the active timeline to t ticks before rendering");
    println!("  --no-timeline        do not play any metadata/timelineControl timeline");
    println!("  --set name=value     submit an immediate variable write before rendering");
    println!("  --set-timed name=value,time,easing");
    println!("                       submit a timed variable write in 1/60s ticks");
    println!("  --control-variable <name>");
    println!("                       use A/D/R/K keys to test SetVariable submission");
    println!("  --dump-frames <dir> export 30 software-composited runtime frames and exit");
    println!("  --debug-frames      print first 30 runtime frame diagnostics while playing");
    println!();
    println!("optional wrapper/decrypt:");
    println!("  --no-mdf             do not unwrap MDF zlib containers");
    println!("  --no-lz4             do not unwrap LZ4 frame containers");
    println!("  --key <u32>          Emote key DWORD; fixed seeds are 0x075BCD15, 0x159A55E5, 0x1F123BB5, key, 0, 0");
    println!();
    println!("without --input, the player shows an internal vertex-pipeline preview mesh");
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    loaded_model: Option<LoadedModel>,
    occluded: bool,
    last_frame: Option<Instant>,
    control_variable: Option<String>,
    egui_ctx: egui::Context,
    egui_winit: Option<EguiWinitState>,
    ui_state: UiState,
}

fn create_egui_context() -> egui::Context {
    let ctx = egui::Context::default();
    install_embedded_egui_font(&ctx);
    ctx
}

fn install_embedded_egui_font(ctx: &egui::Context) {
    let font_bytes = ELUNA_EMBEDDED_DEFAULT_FONT;

    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        EGUI_DEFAULT_FONT_NAME.to_owned(),
        egui::FontData::from_static(font_bytes).into(),
    );

    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, EGUI_DEFAULT_FONT_NAME.to_owned());
    }

    ctx.set_fonts(fonts);
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            gpu: None,
            loaded_model: None,
            occluded: false,
            last_frame: None,
            control_variable: None,
            egui_ctx: create_egui_context(),
            egui_winit: None,
            ui_state: UiState::default(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let title = if self.loaded_model.is_some() {
            "eluna_player - dynamic Emote preview"
        } else {
            "eluna_player - vertex preview"
        };

        let window = match event_loop.create_window(Window::default_attributes().with_title(title))
        {
            Ok(window) => Arc::new(window),
            Err(err) => {
                eprintln!("failed to create window: {err}");
                event_loop.exit();
                return;
            }
        };

        let display_handle = event_loop.owned_display_handle();
        let gpu_result = catch_unwind(AssertUnwindSafe(|| {
            pollster::block_on(GpuState::new(
                window.clone(),
                display_handle,
                self.loaded_model.as_ref(),
            ))
        }));
        let gpu = match gpu_result {
            Ok(Ok(gpu)) => gpu,
            Ok(Err(err)) => {
                eprintln!("failed to initialize wgpu: {err}");
                event_loop.exit();
                return;
            }
            Err(_) => {
                eprintln!("failed to initialize wgpu: initialization panicked; run with RUST_BACKTRACE=1 for the original wgpu validation location");
                event_loop.exit();
                return;
            }
        };

        let egui_winit = EguiWinitState::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            Some(2048),
        );
        self.egui_winit = Some(egui_winit);

        window.request_redraw();

        self.window = Some(window);
        self.gpu = Some(gpu);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref() else {
            return;
        };

        // Pass events to egui first; if consumed, skip game input handling.
        if let Some(egui_winit) = self.egui_winit.as_mut() {
            let response = egui_winit.on_window_event(window, &event);
            if response.repaint {
                window.request_redraw();
            }
            if response.consumed {
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if handle_keyboard(
                    event_loop,
                    &event,
                    self.loaded_model.as_mut(),
                    self.control_variable.as_deref(),
                    window,
                ) {
                    return;
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size);
                    window.request_redraw();
                }
            }
            WindowEvent::Occluded(occluded) => {
                self.occluded = occluded;
                if !occluded {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if self.occluded {
                    return;
                }
                let now = Instant::now();
                if let Some(model) = self.loaded_model.as_mut() {
                    if let Some(prev) = self.last_frame.replace(now) {
                        let raw_ms = (now - prev).as_secs_f32() * 1000.0;
                        let delta_ticks =
                            milliseconds_to_emote_ticks(raw_ms.min(EMOTE_UPDATE_MS_CAP))
                                * self.ui_state.playback_speed;
                        let previous_variables = player_variable_values(&model.player);
                        let previous_scene = model.scene.clone();
                        if self.ui_state.step_update {
                            if self.ui_state.dirty || model.player.is_modified() {
                                model.player.step();
                                model.player.pass();
                            }
                        } else {
                            model.player.progress_ticks_without_physics(delta_ticks);
                        }
                        match rebuild_loaded_model_scene(model, delta_ticks) {
                            Ok(()) => {
                                let variable_values = player_variable_values(&model.player);
                                if model.debug_frames {
                                    if let Some(motion) = model.active_motion.clone() {
                                        debug_runtime_frame(
                                            model,
                                            &motion,
                                            delta_ticks,
                                            &previous_variables,
                                            &previous_scene,
                                            &model.scene.clone(),
                                            &variable_values,
                                        );
                                    }
                                }
                                if let Some(gpu) = self.gpu.as_mut() {
                                    if let Err(err) = gpu.update_model_scene(&model.scene) {
                                        eprintln!("scene update error: {err}");
                                    }
                                }
                                model.player.clear_modified();
                            }
                            Err(err) => eprintln!("motion scene error: {err}"),
                        }
                    }
                } else {
                    self.last_frame = Some(now);
                }
                let egui_output = if let Some(egui_winit) = self.egui_winit.as_mut() {
                    let raw_input = egui_winit.take_egui_input(window);
                    let full_output = self.egui_ctx.run(raw_input, |ctx| {
                        build_ui(ctx, self.loaded_model.as_mut(), &mut self.ui_state);
                    });
                    egui_winit.handle_platform_output(window, full_output.platform_output);
                    let paint_jobs = self
                        .egui_ctx
                        .tessellate(full_output.shapes, full_output.pixels_per_point);
                    Some(EguiOutput {
                        paint_jobs,
                        textures_delta: full_output.textures_delta,
                        pixels_per_point: full_output.pixels_per_point,
                    })
                } else {
                    None
                };
                if self.ui_state.dirty {
                    self.ui_state.dirty = false;
                    if let Some(model) = self.loaded_model.as_mut() {
                        if let Err(err) = rebuild_loaded_model_scene(model, 0.0) {
                            eprintln!("ui scene update error: {err}");
                        } else if let Some(gpu) = self.gpu.as_mut() {
                            if let Err(err) = gpu.update_model_scene(&model.scene) {
                                eprintln!("ui gpu update error: {err}");
                            }
                            model.player.clear_modified();
                        } else {
                            model.player.clear_modified();
                        }
                    }
                }
                let player = self.loaded_model.as_ref().map(|model| &model.player);
                if let Some(gpu) = self.gpu.as_mut() {
                    if let Err(err) = gpu.render(window.clone(), player, egui_output) {
                        eprintln!("render error: {err}");
                    }
                }
                window.request_redraw();
            }
            _ => {}
        }
    }
}

fn handle_keyboard(
    event_loop: &ActiveEventLoop,
    event: &KeyEvent,
    loaded_model: Option<&mut LoadedModel>,
    control_variable: Option<&str>,
    window: &Window,
) -> bool {
    if event.state != ElementState::Pressed {
        return false;
    }

    if matches!(&event.logical_key, Key::Named(NamedKey::Escape)) {
        event_loop.exit();
        return true;
    }

    let Some(model) = loaded_model else {
        return false;
    };

    match &event.logical_key {
        Key::Named(NamedKey::ArrowLeft) => {
            let [x, y] = model.player.coord();
            model.player.set_coord(x - 10.0, y);
            window.request_redraw();
            true
        }
        Key::Named(NamedKey::ArrowRight) => {
            let [x, y] = model.player.coord();
            model.player.set_coord(x + 10.0, y);
            window.request_redraw();
            true
        }
        Key::Named(NamedKey::ArrowUp) => {
            let [x, y] = model.player.coord();
            model.player.set_coord(x, y - 10.0);
            window.request_redraw();
            true
        }
        Key::Named(NamedKey::ArrowDown) => {
            let [x, y] = model.player.coord();
            model.player.set_coord(x, y + 10.0);
            window.request_redraw();
            true
        }
        Key::Named(NamedKey::Space) => {
            if model.player.is_shown() {
                model.player.hide();
            } else {
                model.player.show();
            }
            window.request_redraw();
            true
        }
        Key::Character(text) => match text.as_str() {
            "+" | "=" => {
                model.player.set_scale(model.player.scale() * 1.05);
                window.request_redraw();
                true
            }
            "-" | "_" => {
                model
                    .player
                    .set_scale((model.player.scale() / 1.05).max(0.01));
                window.request_redraw();
                true
            }
            "q" | "Q" => {
                model.player.set_rot(model.player.rot() - 0.05);
                window.request_redraw();
                true
            }
            "e" | "E" => {
                model.player.set_rot(model.player.rot() + 0.05);
                window.request_redraw();
                true
            }
            "a" | "A" => {
                if let Some(name) = control_variable {
                    let value = model.player.variable_value(name).unwrap_or(0.0) - 0.1;
                    model.player.set_variable(name, value);
                    println!("set_variable {name}={value:.3}");
                    window.request_redraw();
                    true
                } else {
                    false
                }
            }
            "d" | "D" => {
                if let Some(name) = control_variable {
                    let value = model.player.variable_value(name).unwrap_or(0.0) + 0.1;
                    model.player.set_variable(name, value);
                    println!("set_variable {name}={value:.3}");
                    window.request_redraw();
                    true
                } else {
                    false
                }
            }
            "r" | "R" => {
                if let Some(name) = control_variable {
                    model.player.set_variable(name, 0.0);
                    println!("set_variable {name}=0.000");
                    window.request_redraw();
                    true
                } else {
                    false
                }
            }
            "k" | "K" => {
                println!(
                    "player: shown={} coord={:?} scale={} rot={} variables={} pending_writes={}",
                    model.player.is_shown(),
                    model.player.coord(),
                    model.player.scale(),
                    model.player.rot(),
                    model.player.variables().len(),
                    model.player.pending_writes().len()
                );
                if let Some(name) = control_variable {
                    println!("variable {name}={:?}", model.player.variable_value(name));
                }
                true
            }
            _ => false,
        },
        _ => false,
    }
}

fn debug_runtime_frame(
    model: &mut LoadedModel,
    motion: &str,
    delta_ticks: f32,
    previous_variables: &BTreeMap<String, f32>,
    previous_scene: &EmoteStaticScene,
    scene: &EmoteStaticScene,
    variables: &BTreeMap<String, f32>,
) {
    if model.debug_frame_count >= 30 {
        return;
    }
    model.debug_frame_count += 1;

    let changed_variables = variables
        .iter()
        .filter(|(name, value)| {
            previous_variables
                .get(*name)
                .map(|prev| (*prev - **value).abs() > 0.0001)
                .unwrap_or(true)
        })
        .count();

    let previous_by_path: BTreeMap<&str, &EmoteStaticSprite> = previous_scene
        .sprites
        .iter()
        .map(|sprite| (sprite.draw_frame_info.path.as_str(), sprite))
        .collect();
    let mut changed_layers = Vec::new();
    let mut changed_meshes = Vec::new();
    let mut first_vertex_delta = None;
    for sprite in &scene.sprites {
        let Some(prev) = previous_by_path
            .get(sprite.draw_frame_info.path.as_str())
            .copied()
        else {
            changed_layers.push(sprite.draw_frame_info.path.clone());
            continue;
        };
        let layer_changed = (prev.opacity - sprite.opacity).abs() > 0.0001
            || (prev.center_x - sprite.center_x).abs() > 0.0001
            || (prev.center_y - sprite.center_y).abs() > 0.0001
            || prev.world_transform != sprite.world_transform
            || prev.visible != sprite.visible;
        if layer_changed {
            changed_layers.push(sprite.draw_frame_info.path.clone());
        }
        if prev.mesh != sprite.mesh {
            changed_meshes.push(sprite.draw_frame_info.path.clone());
        }
        if first_vertex_delta.is_none() {
            let prev_vertices = sprite_vertices(prev);
            let next_vertices = sprite_vertices(sprite);
            if let (Some(a), Some(b)) = (prev_vertices.first(), next_vertices.first()) {
                if (a.position[0] - b.position[0]).abs() > 0.0001
                    || (a.position[1] - b.position[1]).abs() > 0.0001
                {
                    first_vertex_delta =
                        Some((a.position, b.position, sprite.draw_frame_info.path.clone()));
                }
            }
        }
    }

    let draw_layer_count = scene
        .sprites
        .iter()
        .filter(|sprite| sprite.visible && sprite.opacity > 0.0)
        .count();
    let draw_mesh_count = scene
        .sprites
        .iter()
        .filter(|sprite| sprite.visible && sprite.opacity > 0.0 && sprite.mesh.is_some())
        .count();
    let face_related: Vec<(usize, &EmoteStaticSprite)> = scene
        .sprites
        .iter()
        .enumerate()
        .filter(|(_, sprite)| is_face_related_sprite(sprite))
        .collect();
    let face_submitted: Vec<(usize, &EmoteStaticSprite)> = face_related
        .iter()
        .copied()
        .filter(|(_, sprite)| {
            sprite.visible
                && sprite.opacity > 0.0
                && (sprite.texture_resource_index as usize) < model.psb.resources.len()
                && sprite.draw_frame_info.pass != EmoteDrawPass::MaskGeneration
        })
        .collect();
    let face_skipped = face_related.len().saturating_sub(face_submitted.len());
    let missing_texture_refs = scene
        .sprites
        .iter()
        .filter(|sprite| {
            sprite.visible
                && sprite.opacity > 0.0
                && sprite.texture_resource_index as usize >= model.psb.resources.len()
        })
        .count();
    let first_draw = scene
        .sprites
        .iter()
        .find(|sprite| sprite.visible && sprite.opacity > 0.0);
    eprintln!(
        "emote frame {} motion='{}' dt={:.3} motion_time={:.3} frame={:?}->{:?} t={:.3} vars_changed={} layers_changed={} meshes_changed={} draw_layers={} draw_meshes={} face_submitted={} face_skipped={} textures={} missing_texture_refs={}",
        model.debug_frame_count,
        motion,
        delta_ticks,
        model.player.elapsed_ticks(),
        first_draw.and_then(|sprite| sprite.draw_frame_info.frame_index),
        first_draw.and_then(|sprite| sprite.draw_frame_info.next_frame_index),
        first_draw.map(|sprite| sprite.draw_frame_info.interpolation_t).unwrap_or(0.0),
        changed_variables,
        changed_layers.len(),
        changed_meshes.len(),
        draw_layer_count,
        draw_mesh_count,
        face_submitted.len(),
        face_skipped,
        model.schema.textures.len(),
        missing_texture_refs,
    );
    for sprite in scene
        .sprites
        .iter()
        .filter(|sprite| sprite.visible && sprite.opacity > 0.0)
        .take(5)
    {
        eprintln!(
            "  draw path='{}' label={:?} opacity={:.3} control={:?} value={:?} local_time={:?} selector arm_type={:?}",
            sprite.draw_frame_info.path,
            sprite.label,
            sprite.opacity,
            sprite.draw_frame_info.control_parameter,
            sprite.draw_frame_info.control_value,
            sprite.draw_frame_info.local_time_ticks,
            variables.get("arm_type"),
        );
    }
    // Frame 1 only: full draw-order diagnostic around face/head/mouth/nose layers.
    // This identifies which layer is the "large pink block" covering the face
    // and what its z-value / draw_index context is relative to the face layers.
    if model.debug_frame_count == 1 {
        let all_sprites: Vec<(usize, &EmoteStaticSprite)> =
            scene.sprites.iter().enumerate().collect();

        // Print ALL visible sprites in draw order with abbreviated info
        eprintln!(
            "=== FULL DRAW ORDER (all {} sprites, {} visible+opaque) ===",
            all_sprites.len(),
            draw_layer_count
        );
        for (order, sprite) in &all_sprites {
            let vis = sprite.visible && sprite.opacity > 0.0;
            let face = is_face_related_sprite(sprite);
            let large = sprite.width > 200.0 || sprite.height > 200.0;
            let flag = match (vis, face, large) {
                (false, _, _) => "HIDDEN",
                (true, true, _) => "FACE",
                (true, false, true) => "LARGE",
                _ => "draw",
            };
            eprintln!(
                "  [{:>3}] z={:>6.2} di={:>3} pass={:<24} stencil={} masks={} op={:.2} vis={} tex={} \
                 size={:.0}x{:.0} path='{}' label={:?} {}",
                order,
                sprite.z,
                sprite.draw_frame_info.draw_index,
                format!("{:?}", sprite.draw_frame_info.pass),
                sprite.draw_frame_info.stencil_type,
                sprite.draw_frame_info.stencil_composite_mask_layer_list.len(),
                sprite.opacity,
                sprite.visible as u8,
                sprite.texture_resource_index,
                sprite.width,
                sprite.height,
                sprite.draw_frame_info.path,
                sprite.draw_frame_info.layer_label,
                flag,
            );
        }

        // Per face layer: show the 4 sprites immediately above it in draw order (drawn later = on top)
        eprintln!("=== FACE LAYER CONTEXT (layers drawn on top of each face layer) ===");
        for (order, sprite) in face_related
            .iter()
            .copied()
            .filter(|(_, s)| s.visible && s.opacity > 0.0)
        {
            let above: Vec<_> = all_sprites
                .iter()
                .filter(|(o, s)| {
                    *o > order
                        && s.visible
                        && s.opacity > 0.0
                        && s.draw_frame_info.pass != EmoteDrawPass::MaskGeneration
                })
                .take(4)
                .collect();
            eprintln!(
                "  FACE [{}] z={:.2} di={} pass={:?} stencil={} size={:.0}x{:.0} op={:.2} path='{}' label={:?}",
                order, sprite.z, sprite.draw_frame_info.draw_index,
                sprite.draw_frame_info.pass, sprite.draw_frame_info.stencil_type,
                sprite.width, sprite.height, sprite.opacity,
                sprite.draw_frame_info.path, sprite.draw_frame_info.layer_label,
            );
            for (above_order, above_sprite) in &above {
                let face_tag = if is_face_related_sprite(above_sprite) {
                    "FACE"
                } else {
                    ""
                };
                eprintln!(
                    "    ABOVE[{}] z={:.2} di={} pass={:?} stencil={} size={:.0}x{:.0} op={:.2} tex={} path='{}' {}",
                    above_order, above_sprite.z, above_sprite.draw_frame_info.draw_index,
                    above_sprite.draw_frame_info.pass, above_sprite.draw_frame_info.stencil_type,
                    above_sprite.width, above_sprite.height, above_sprite.opacity,
                    above_sprite.texture_resource_index,
                    above_sprite.draw_frame_info.path, face_tag,
                );
            }
        }
    } else {
        // Subsequent frames: compact face summary only
        for (draw_order, sprite) in face_related {
            let skipped_reason =
                if sprite.texture_resource_index as usize >= model.psb.resources.len() {
                    "missing-texture"
                } else if !sprite.visible {
                    "visibility=false"
                } else if sprite.opacity <= 0.0 {
                    "opacity=0"
                } else if sprite.draw_frame_info.pass == EmoteDrawPass::MaskGeneration {
                    "mask-generation"
                } else {
                    "submitted"
                };
            eprintln!(
                "  face order={} di={} z={:.2} pass={:?} stencil={} size={:.0}x{:.0} op={:.2} reason={} path='{}'",
                draw_order, sprite.draw_frame_info.draw_index, sprite.z,
                sprite.draw_frame_info.pass, sprite.draw_frame_info.stencil_type,
                sprite.width, sprite.height, sprite.opacity, skipped_reason,
                sprite.draw_frame_info.path,
            );
        }
    }
    for path in changed_layers.iter().take(5) {
        eprintln!("  changed layer: {path}");
    }
    for path in changed_meshes.iter().take(5) {
        eprintln!("  changed mesh: {path}");
    }
    if let Some((before, after, path)) = first_vertex_delta {
        eprintln!("  vertex delta path='{path}' before={before:?} after={after:?}");
    } else {
        eprintln!("  vertex delta: none detected");
    }

    // Physics variable outputs
    let physics_vars = [
        "bust_LR",
        "bust_UD",
        "bust_LR_spare",
        "bust_UD_spare",
        "hair_LR_front",
        "hair_LR_M_front",
        "hair_UD_front",
        "hair_LR_side_L",
        "hair_LR_M_side_L",
        "hair_UD_side_L",
        "hair_LR_side_R",
        "hair_LR_M_side_R",
        "hair_UD_side_R",
    ];
    let phys_out: Vec<String> = physics_vars
        .iter()
        .filter_map(|name| variables.get(*name).map(|v| format!("{}={:.3}", name, v)))
        .collect();
    if !phys_out.is_empty() {
        eprintln!("  physics: {}", phys_out.join(", "));
    }
}

fn is_face_related_sprite(sprite: &EmoteStaticSprite) -> bool {
    let path = sprite.draw_frame_info.path.as_str();
    path.contains("face")
        || path.contains("eye")
        || path.contains("mouth")
        || path.contains("eyebrow")
        || path.contains("head")
        || path.contains("nose")
        || path.contains("目")
        || path.contains("眉")
        || path.contains("口")
        || path.contains("鼻")
        || path.contains("輪郭")
        || path.contains("頬")
        || path.contains("頭")
        || path.contains("表情")
}

struct GpuState {
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    mask_pipeline: wgpu::RenderPipeline,
    masked_pipeline: wgpu::RenderPipeline,
    mask_bind_group_layout: wgpu::BindGroupLayout,
    mask_sampler: wgpu::Sampler,
    mask_format: wgpu::TextureFormat,
    mask_size: (u32, u32),
    mask_targets: BTreeMap<u32, MaskCacheEntry>,
    transform_buffer: wgpu::Buffer,
    transform_bind_group: wgpu::BindGroup,
    textures: Vec<GpuTexture>,
    model_texture_slots: BTreeMap<u32, usize>,
    draws: Vec<GpuDraw>,
    bounds: EmoteSceneBounds,
    egui_renderer: EguiRenderer,
    warned_missing_egui_texture: bool,
}

struct GpuTexture {
    _texture: wgpu::Texture,
    _view: wgpu::TextureView,
    _sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
}

struct GpuDraw {
    texture_slot: usize,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    pass: EmoteDrawPass,
    mask_reference: u32,
    parent_mask_reference: u32,
}

struct MaskCacheEntry {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

fn create_alpha_mask_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_alpha_mask_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn create_color_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
    label: &'static str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[GpuSpriteVertex::layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

impl GpuState {
    async fn new(
        window: Arc<Window>,
        display_handle: OwnedDisplayHandle,
        loaded_model: Option<&LoadedModel>,
    ) -> Result<Self, Box<dyn Error>> {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(display_handle)),
        );
        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("eluna device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await?;

        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| {
                IoError::new(
                    ErrorKind::Other,
                    "surface is not supported by the selected adapter",
                )
            })?;
        config.desired_maximum_frame_latency = 2;
        eprintln!("eluna surface format: {:?}", config.format);
        surface.configure(&device, &config);

        let transform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("eluna transform bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("eluna texture bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let mask_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("eluna alpha mask bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("eluna sprite shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let masked_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("eluna masked sprite shader"),
            source: wgpu::ShaderSource::Wgsl(MASKED_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("eluna sprite pipeline layout"),
            bind_group_layouts: &[
                Some(&transform_bind_group_layout),
                Some(&texture_bind_group_layout),
            ],
            immediate_size: 0,
        });
        let masked_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("eluna masked sprite pipeline layout"),
                bind_group_layouts: &[
                    Some(&transform_bind_group_layout),
                    Some(&texture_bind_group_layout),
                    Some(&mask_bind_group_layout),
                ],
                immediate_size: 0,
            });

        // Mask compositing follows the official JS Tyrano renderer (ALPHA mask
        // mode, stencil disabled, depth disabled).  Each parent mask reference
        // owns its own Rgba8Unorm alpha texture; masked color items sample the
        // corresponding texture instead of using a shared stencil buffer.
        let mask_format = wgpu::TextureFormat::Rgba8Unorm;
        let pipeline = create_color_pipeline(
            &device,
            &pipeline_layout,
            &shader,
            config.format,
            "eluna sprite pipeline",
        );
        let mask_pipeline = create_color_pipeline(
            &device,
            &pipeline_layout,
            &shader,
            mask_format,
            "eluna alpha mask writer pipeline",
        );
        let masked_pipeline = create_color_pipeline(
            &device,
            &masked_pipeline_layout,
            &masked_shader,
            config.format,
            "eluna masked sprite pipeline",
        );
        let mask_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("eluna alpha mask sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let transform = TransformUniform {
            center: [0.0, 0.0],
            viewport_scale: [1.0, 1.0],
            player_coord: [0.0, 0.0],
            player_scale: 1.0,
            player_cos: 1.0,
            player_sin: 0.0,
            _pad: 0.0,
        };
        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("eluna transform uniform"),
            contents: bytemuck::bytes_of(&transform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eluna transform bind group"),
            layout: &transform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: transform_buffer.as_entire_binding(),
            }],
        });

        let egui_renderer = EguiRenderer::new(&device, config.format, RendererOptions::default());

        let (textures, model_texture_slots, draws, bounds) = if let Some(model) = loaded_model {
            build_model_assets(
                &device,
                &queue,
                &texture_bind_group_layout,
                &model.normalized_data,
                &model.psb,
                &model.schema,
                &model.scene,
            )?
        } else {
            let (textures, draws, bounds) =
                build_preview_draws(&device, &queue, &texture_bind_group_layout)?;
            (textures, BTreeMap::new(), draws, bounds)
        };

        let state = Self {
            _instance: instance,
            surface,
            _adapter: adapter,
            device,
            queue,
            config,
            pipeline,
            mask_pipeline,
            masked_pipeline,
            mask_bind_group_layout,
            mask_sampler,
            mask_format,
            mask_size: (width, height),
            mask_targets: BTreeMap::new(),
            transform_buffer,
            transform_bind_group,
            textures,
            model_texture_slots,
            draws,
            bounds,
            egui_renderer,
            warned_missing_egui_texture: false,
        };
        state.update_transform(None);
        Ok(state)
    }

    fn update_model_scene(&mut self, scene: &EmoteStaticScene) -> Result<(), Box<dyn Error>> {
        let (draws, bounds) = build_scene_draws(&self.device, scene, &self.model_texture_slots)?;
        self.draws = draws;
        self.bounds = bounds;
        Ok(())
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        self.config.width = size.width.max(1);
        self.config.height = size.height.max(1);
        self.surface.configure(&self.device, &self.config);
        // Mask textures are sized to match the surface so the masked-color
        // shader's screen-space mask UV matches the parent mask coverage.
        // Drop the cache; entries are reallocated lazily before the next pass.
        self.mask_size = (self.config.width, self.config.height);
        self.mask_targets.clear();
        self.update_transform(None);
    }

    fn ensure_mask_targets<I>(&mut self, references: I)
    where
        I: IntoIterator<Item = u32>,
    {
        use std::collections::BTreeSet;
        let needed: BTreeSet<u32> = references.into_iter().filter(|&r| r != 0).collect();
        let (width, height) = self.mask_size;
        // Drop entries for references that are no longer in use; their textures
        // would otherwise carry stale mask coverage into a future frame.
        self.mask_targets.retain(|key, _| needed.contains(key));
        for reference in needed {
            if self.mask_targets.contains_key(&reference) {
                continue;
            }
            let label = format!("eluna alpha mask #{}", reference);
            let (texture, view) =
                create_alpha_mask_target(&self.device, width, height, self.mask_format, &label);
            let bind_group = create_alpha_mask_bind_group(
                &self.device,
                &self.mask_bind_group_layout,
                &view,
                &self.mask_sampler,
                &format!("eluna alpha mask bind group #{}", reference),
            );
            self.mask_targets.insert(
                reference,
                MaskCacheEntry {
                    _texture: texture,
                    view,
                    bind_group,
                },
            );
        }
    }

    fn update_transform(&self, player: Option<&ElunaPlayer>) {
        let center = self.bounds.center();
        let bounds_w = self.bounds.width().max(1.0);
        let bounds_h = self.bounds.height().max(1.0);
        let viewport_w = self.config.width.max(1) as f32;
        let viewport_h = self.config.height.max(1) as f32;
        let fit = 0.9 * (viewport_w / bounds_w).min(viewport_h / bounds_h);
        let (player_coord, player_scale, player_rot) = if let Some(player) = player {
            (player.coord(), player.scale(), player.rot())
        } else {
            ([0.0, 0.0], 1.0, 0.0)
        };
        let uniform = TransformUniform {
            center,
            viewport_scale: [fit * 2.0 / viewport_w, fit * 2.0 / viewport_h],
            player_coord,
            player_scale,
            player_cos: player_rot.cos(),
            player_sin: player_rot.sin(),
            _pad: 0.0,
        };
        self.queue
            .write_buffer(&self.transform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    fn render(
        &mut self,
        window: Arc<Window>,
        player: Option<&ElunaPlayer>,
        egui_output: Option<EguiOutput>,
    ) -> Result<(), Box<dyn Error>> {
        use wgpu::CurrentSurfaceTexture;

        self.update_transform(player);

        // Apply egui texture updates before acquiring the surface texture.
        // winit/wgpu may transiently return Timeout/Occluded/Lost during the first
        // frames. If the first full font-atlas allocation is dropped in that path,
        // egui-wgpu later receives only a partial update and either panics or renders
        // no visible UI. Texture uploads do not depend on the current surface frame,
        // so keep them outside the surface-acquire early-return path.
        let mut egui_textures_to_free = Vec::new();
        if let Some(output) = egui_output.as_ref() {
            for (id, delta) in output
                .textures_delta
                .set
                .iter()
                .filter(|(_, delta)| delta.pos.is_none())
            {
                self.egui_renderer
                    .update_texture(&self.device, &self.queue, *id, delta);
            }
            for (id, delta) in output
                .textures_delta
                .set
                .iter()
                .filter(|(_, delta)| delta.pos.is_some())
            {
                if self.egui_renderer.texture(id).is_some() {
                    self.egui_renderer
                        .update_texture(&self.device, &self.queue, *id, delta);
                } else {
                    if !self.warned_missing_egui_texture {
                        eprintln!(
                            "egui warning: skipped partial texture update for unallocated texture {:?}; requesting repaint",
                            id
                        );
                        self.warned_missing_egui_texture = true;
                    }
                    window.request_redraw();
                }
            }
            egui_textures_to_free.extend(output.textures_delta.free.iter().cloned());
        }

        let frame = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(frame) | CurrentSurfaceTexture::Suboptimal(frame) => {
                frame
            }
            CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => return Ok(()),
            CurrentSurfaceTexture::Outdated | CurrentSurfaceTexture::Lost => {
                let size = window.inner_size();
                self.resize(size);
                return Ok(());
            }
            CurrentSurfaceTexture::Validation => {
                return Err(Box::new(IoError::new(
                    ErrorKind::Other,
                    "surface validation error",
                )));
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("eluna sprite encoder"),
            });

        let background = wgpu::Color {
            r: 0.05,
            g: 0.05,
            b: 0.06,
            a: 1.0,
        };
        let player_visible = player.map(|player| player.is_shown()).unwrap_or(true);

        // Collect the parent mask references this frame uses so each one gets
        // its own Rgba8Unorm alpha texture.  Sharing one texture across groups
        // would erase coverage that a later masked-color draw still needs.
        let mut mask_refs_needed = std::collections::BTreeSet::<u32>::new();
        if player_visible {
            for draw in &self.draws {
                if draw.parent_mask_reference != 0 {
                    mask_refs_needed.insert(draw.parent_mask_reference);
                }
            }
        }
        self.ensure_mask_targets(mask_refs_needed.iter().copied());

        if player_visible {
            // Build each parent mask reference's alpha texture once per frame.
            // Mask source items keep their drawFrameInfo emission order; we
            // simply filter by mask_reference so the texture contains the alpha
            // coverage for that reference and nothing else.
            for &reference in &mask_refs_needed {
                let Some(entry) = self.mask_targets.get(&reference) else {
                    continue;
                };
                let mut mask_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("eluna alpha mask reference pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &entry.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                });
                mask_pass.set_pipeline(&self.mask_pipeline);
                mask_pass.set_bind_group(0, &self.transform_bind_group, &[]);
                for draw in &self.draws {
                    if draw.mask_reference != reference {
                        continue;
                    }
                    if draw.pass != EmoteDrawPass::MaskGeneration
                        && draw.pass != EmoteDrawPass::StencilCompositeMask
                    {
                        continue;
                    }
                    let Some(texture) = self.textures.get(draw.texture_slot) else {
                        continue;
                    };
                    mask_pass.set_bind_group(1, &texture.bind_group, &[]);
                    mask_pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                    mask_pass.draw(0..draw.vertex_count, 0..1);
                }
            }
        }

        // Single color pass consumes the drawFrameInfo stream in original order.
        // Pipeline switches per draw based on whether the item samples a parent
        // mask; we never resort and never reuse a shared stencil reference.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("eluna color drawFrameInfo pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(background),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            if player_visible {
                #[derive(Copy, Clone, PartialEq, Eq)]
                enum Mode {
                    None,
                    Color,
                    Masked,
                }
                let mut mode = Mode::None;
                for draw in &self.draws {
                    if draw.pass == EmoteDrawPass::MaskGeneration
                        || draw.pass == EmoteDrawPass::StencilCompositeMask
                    {
                        continue;
                    }
                    let Some(texture) = self.textures.get(draw.texture_slot) else {
                        continue;
                    };
                    if draw.parent_mask_reference != 0 {
                        let Some(mask_entry) = self.mask_targets.get(&draw.parent_mask_reference)
                        else {
                            continue;
                        };
                        if mode != Mode::Masked {
                            pass.set_pipeline(&self.masked_pipeline);
                            // Re-bind every group after a pipeline switch.  The
                            // masked pipeline layout has 3 bind groups while the
                            // normal pipeline layout has 2; wgpu treats the
                            // groups beyond the previous pipeline's layout as
                            // unset, and re-binding 0 is cheap insurance against
                            // any cross-pipeline state aliasing.
                            pass.set_bind_group(0, &self.transform_bind_group, &[]);
                            mode = Mode::Masked;
                        }
                        pass.set_bind_group(2, &mask_entry.bind_group, &[]);
                        pass.set_bind_group(1, &texture.bind_group, &[]);
                    } else {
                        if mode != Mode::Color {
                            pass.set_pipeline(&self.pipeline);
                            pass.set_bind_group(0, &self.transform_bind_group, &[]);
                            mode = Mode::Color;
                        }
                        pass.set_bind_group(1, &texture.bind_group, &[]);
                    }
                    pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                    pass.draw(0..draw.vertex_count, 0..1);
                }
            }
        }

        // Egui overlay pass. Texture uploads were already processed before
        // surface acquisition above, so this pass only updates buffers and draws.
        if let Some(output) = egui_output {
            let screen_desc = ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: output.pixels_per_point,
            };
            self.egui_renderer.update_buffers(
                &self.device,
                &self.queue,
                &mut encoder,
                &output.paint_jobs,
                &screen_desc,
            );
            {
                let mut egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                });
                // egui_wgpu requires 'static lifetime; forget_lifetime is safe here
                // because the pass is dropped before encoder.finish()
                let mut egui_pass = egui_pass.forget_lifetime();
                self.egui_renderer
                    .render(&mut egui_pass, &output.paint_jobs, &screen_desc);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        for id in &egui_textures_to_free {
            self.egui_renderer.free_texture(id);
        }
        window.pre_present_notify();
        frame.present();
        Ok(())
    }
}

fn build_model_assets(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_layout: &wgpu::BindGroupLayout,
    data: &[u8],
    psb: &PsbFile,
    schema: &EmoteModelSchema,
    scene: &EmoteStaticScene,
) -> Result<
    (
        Vec<GpuTexture>,
        BTreeMap<u32, usize>,
        Vec<GpuDraw>,
        EmoteSceneBounds,
    ),
    Box<dyn Error>,
> {
    let mut texture_slots = BTreeMap::<u32, usize>::new();
    let mut textures = Vec::<GpuTexture>::new();

    for texture_source in schema.textures.values() {
        if texture_slots.contains_key(&texture_source.resource_index) {
            continue;
        }
        let bytes = psb
            .resource_bytes(data, texture_source.resource_index as usize)
            .ok_or_else(|| {
                IoError::new(
                    ErrorKind::InvalidData,
                    "texture resource index out of range",
                )
            })?;
        let decoded = decode_texture_rgba(
            bytes,
            texture_source.width,
            texture_source.height,
            texture_source.format.as_deref(),
            texture_source.compress.as_deref(),
            texture_source.bit_count,
            schema.spec.as_deref(),
        )?;
        let texture = create_texture_bind_group(
            device,
            queue,
            texture_layout,
            &decoded.rgba,
            decoded.width,
            decoded.height,
            &format!("{}#{}", texture_source.name, texture_source.resource_index),
        );
        let slot = textures.len();
        textures.push(texture);
        texture_slots.insert(texture_source.resource_index, slot);
    }

    let (draws, bounds) = build_scene_draws(device, scene, &texture_slots)?;
    Ok((textures, texture_slots, draws, bounds))
}

fn build_scene_draws(
    device: &wgpu::Device,
    scene: &EmoteStaticScene,
    texture_slots: &BTreeMap<u32, usize>,
) -> Result<(Vec<GpuDraw>, EmoteSceneBounds), Box<dyn Error>> {
    let visible_sprites: Vec<&EmoteStaticSprite> = scene
        .sprites
        .iter()
        .filter(|sprite| sprite.visible && sprite.opacity > 0.0)
        .collect();

    let mut path_to_sprite = BTreeMap::<String, &EmoteStaticSprite>::new();
    for sprite in &visible_sprites {
        path_to_sprite.insert(sprite.draw_frame_info.path.clone(), *sprite);
    }

    // mask_refs keys are OWNER paths read from `scene.composite_mask_owners`.
    // Per sub_103390C0 second pass (lines 407-528), an owner is a layer with
    // `stencilType & 4` set; its `stencilCompositeMaskLayerList` is a layer-
    // local field consumed at THAT owner.  Descendants of the owner inherit a
    // `parent_mask_path` pointing back at the owner — that is the linkage the
    // renderer uses to sample the owner's alpha texture.  Inheritance of the
    // source list through traversal context (the previous behavior) made
    // descendants accidentally become mask owners themselves.
    let mut mask_refs = BTreeMap::<String, u32>::new();
    let mut next_mask_ref = 1u32;
    for owner_path in scene.composite_mask_owners.keys() {
        if mask_refs.contains_key(owner_path) {
            continue;
        }
        mask_refs.insert(owner_path.clone(), next_mask_ref.min(255));
        next_mask_ref = (next_mask_ref + 1).min(255);
    }

    let dump_enabled = !GPU_DRAW_DUMP_DONE.swap(true, Ordering::Relaxed)
        && std::env::var("ELUNA_DUMP_DRAWS")
            .map(|v| v != "0")
            .unwrap_or(true);
    if dump_enabled {
        eprintln!(
            "eluna draw stream dump: sprites_in_scene={} visible={} mask_refs={}",
            scene.sprites.len(),
            path_to_sprite.len(),
            mask_refs.len(),
        );
        for (path, reference) in &mask_refs {
            eprintln!("  mask_ref[{reference}] -> {path}");
        }
    }

    let mut draws = Vec::<GpuDraw>::new();
    let mut dump_rows = Vec::<DumpRow>::new();
    // Phase 1: for every composite-mask owner, emit MaskGeneration draws from
    // each resolved source path.  These draws are filtered out of the final
    // color pass by the renderer (pass == MaskGeneration is skipped); the
    // alpha they accumulate into the owner's per-reference texture is what
    // descendant Filtered draws sample.
    for (owner_path, references) in &scene.composite_mask_owners {
        let Some(&ref_id) = mask_refs.get(owner_path) else {
            continue;
        };
        for target_path in references {
            let Some(mask_sprite) = path_to_sprite.get(target_path).copied() else {
                continue;
            };
            push_gpu_draw(
                device,
                texture_slots,
                &mut draws,
                mask_sprite,
                EmoteDrawPass::MaskGeneration,
                ref_id,
                0,
            )?;
            if dump_enabled {
                dump_rows.push(DumpRow {
                    gpu_index: draws.len() - 1,
                    sprite: mask_sprite,
                    pass: EmoteDrawPass::MaskGeneration,
                    mask_reference: ref_id,
                    parent_mask_reference: 0,
                    is_mask_source: true,
                    composite_owner: Some(owner_path.clone()),
                });
            }
        }
    }
    // Phase 2: emit every visible sprite in drawFrameInfo order.  A sprite is
    // a masked-color item if its `parent_mask_path` resolves to a known
    // composite-mask owner.  The renderer iterates these in this exact order
    // and routes them between the normal and masked pipelines.
    for sprite in visible_sprites {
        let parent_mask_reference = sprite
            .draw_frame_info
            .parent_mask_path
            .as_ref()
            .and_then(|path| mask_refs.get(path).copied())
            .unwrap_or(0);
        push_gpu_draw(
            device,
            texture_slots,
            &mut draws,
            sprite,
            sprite.draw_frame_info.pass,
            0,
            parent_mask_reference,
        )?;
        if dump_enabled {
            let is_mask_source = matches!(
                sprite.draw_frame_info.pass,
                EmoteDrawPass::MaskGeneration | EmoteDrawPass::StencilCompositeMask
            );
            dump_rows.push(DumpRow {
                gpu_index: draws.len() - 1,
                sprite,
                pass: sprite.draw_frame_info.pass,
                mask_reference: 0,
                parent_mask_reference,
                is_mask_source,
                composite_owner: None,
            });
        }
    }

    if dump_enabled {
        // Compute the actual color-pass render-order index for each entry: the
        // renderer iterates self.draws in order, skips mask sources, and issues
        // the remaining items via the color pipeline (normal or masked).  This
        // mirrors the render flow in GpuState::render exactly.
        let mut render_order = 0usize;
        let mut color_pass_items = Vec::<&DumpRow>::new();
        let mut mask_pass_items = Vec::<&DumpRow>::new();
        let mut per_draw_render_order = vec![-1i32; dump_rows.len()];
        for (i, row) in dump_rows.iter().enumerate() {
            if row.is_mask_source {
                mask_pass_items.push(row);
            } else {
                per_draw_render_order[i] = render_order as i32;
                color_pass_items.push(row);
                render_order += 1;
            }
        }
        eprintln!(
            "eluna pass classification: mask_pass_items={} color_pass_items={} total_gpu_draws={}",
            mask_pass_items.len(),
            color_pass_items.len(),
            dump_rows.len(),
        );
        for (i, row) in dump_rows.iter().enumerate() {
            dump_draw_row(row, per_draw_render_order[i]);
        }
        // Audit table: color-pass items in render order with drawFrameInfo
        // index to verify the stream is monotonic in di.  Any non-monotonic
        // step here would indicate that the GPU stream reorders relative to
        // the original drawFrameInfo emission.
        eprintln!("color pass order audit (render_index, drawFrameInfo_index, label):");
        let mut last_di: Option<usize> = None;
        let mut monotonic = true;
        for row in &color_pass_items {
            let di = row.sprite.draw_frame_info.draw_index;
            let render_index = per_draw_render_order[row.gpu_index] as usize;
            let monotonic_flag = match last_di {
                Some(prev) if di < prev => {
                    monotonic = false;
                    "!REORDER"
                }
                _ => "ok",
            };
            eprintln!(
                "  ro={:>3} di={:>3} parent_ref={:>3} {} label={:?}",
                render_index,
                di,
                row.parent_mask_reference,
                monotonic_flag,
                row.sprite.draw_frame_info.layer_label,
            );
            last_di = Some(di);
        }
        eprintln!(
            "color pass order audit: {}",
            if monotonic {
                "PASS: render order is monotonic in drawFrameInfo index"
            } else {
                "FAIL: render order is NOT monotonic"
            }
        );
        eprintln!("mask pass items grouped by reference:");
        for (path, reference) in &mask_refs {
            let sources: Vec<&DumpRow> = mask_pass_items
                .iter()
                .filter(|row| row.mask_reference == *reference)
                .copied()
                .collect();
            eprintln!(
                "  reference={} sources={} owner_path={}",
                reference,
                sources.len(),
                path,
            );
            for row in &sources {
                eprintln!(
                    "    src di={:>3} label={:?} path={}",
                    row.sprite.draw_frame_info.draw_index,
                    row.sprite.draw_frame_info.layer_label,
                    row.sprite.draw_frame_info.path,
                );
            }
        }
    }

    let bounds = scene.bounds.unwrap_or(EmoteSceneBounds {
        min_x: -1.0,
        min_y: -1.0,
        max_x: 1.0,
        max_y: 1.0,
    });
    Ok((draws, bounds))
}

#[derive(Clone)]
struct DumpRow<'a> {
    gpu_index: usize,
    sprite: &'a EmoteStaticSprite,
    pass: EmoteDrawPass,
    mask_reference: u32,
    parent_mask_reference: u32,
    is_mask_source: bool,
    composite_owner: Option<String>,
}

fn dump_draw_row(row: &DumpRow, render_order_index: i32) {
    let sprite = row.sprite;
    let pass_kind = match row.pass {
        EmoteDrawPass::MaskGeneration | EmoteDrawPass::StencilCompositeMask => "mask",
        EmoteDrawPass::Filtered | EmoteDrawPass::Normal => "color",
    };
    let render_order_disp = if render_order_index < 0 {
        "skip".to_owned()
    } else {
        render_order_index.to_string()
    };
    let m = sprite.world_transform;
    // Screen-space bounds: apply the sprite's world transform to its four
    // corners.  Useful to spot off-screen / inverted layers in the audit.
    let local_corners = [
        [sprite.left(), sprite.top()],
        [sprite.right(), sprite.top()],
        [sprite.right(), sprite.bottom()],
        [sprite.left(), sprite.bottom()],
    ];
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for [x, y] in local_corners {
        let wx = m[0] * x + m[1] * y + m[4];
        let wy = m[2] * x + m[3] * y + m[5];
        min_x = min_x.min(wx);
        min_y = min_y.min(wy);
        max_x = max_x.max(wx);
        max_y = max_y.max(wy);
    }
    let mesh_domain = sprite
        .mesh
        .and_then(|mesh| mesh.domain)
        .map(|[x, y, w, h]| format!("domain=({x:.1},{y:.1},{w:.1},{h:.1})"))
        .unwrap_or_else(|| "domain=none".to_owned());
    let mesh_flag = if sprite.mesh.is_some() { "mesh=1" } else { "mesh=0" };
    eprintln!(
        "  draw[{:>3}] di={:>3} ro={:>4} pass={:<22} pass_kind={:<5} mask_ref={:>3} parent_ref={:>3} \
         is_mask_source={} is_color_item={} stencilType={} inheritMask={:?} \
         opacity={:.3} enabled={} tex={} uv=({:.3},{:.3},{:.3},{:.3}) size={:.0}x{:.0} \
         {} {} \
         world=[{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}] \
         screen_bounds=({:.1},{:.1})-({:.1},{:.1}) \
         frame_idx={:?} next_frame={:?} interp_t={:.3} \
         label={:?} path={} composite_owner={:?}",
        row.gpu_index,
        sprite.draw_frame_info.draw_index,
        render_order_disp,
        format!("{:?}", row.pass),
        pass_kind,
        row.mask_reference,
        row.parent_mask_reference,
        row.is_mask_source as u8,
        (!row.is_mask_source) as u8,
        sprite.draw_frame_info.stencil_type,
        sprite.draw_frame_info.inherit_mask,
        sprite.opacity,
        (sprite.visible && sprite.opacity > 0.0) as u8,
        sprite.texture_resource_index,
        sprite.uv_left,
        sprite.uv_top,
        sprite.uv_right,
        sprite.uv_bottom,
        sprite.width,
        sprite.height,
        mesh_flag,
        mesh_domain,
        m[0], m[1], m[2], m[3], m[4], m[5],
        min_x, min_y, max_x, max_y,
        sprite.draw_frame_info.frame_index,
        sprite.draw_frame_info.next_frame_index,
        sprite.draw_frame_info.interpolation_t,
        sprite.draw_frame_info.layer_label,
        sprite.draw_frame_info.path,
        row.composite_owner,
    );
}

fn push_gpu_draw(
    device: &wgpu::Device,
    texture_slots: &BTreeMap<u32, usize>,
    draws: &mut Vec<GpuDraw>,
    sprite: &EmoteStaticSprite,
    pass: EmoteDrawPass,
    mask_reference: u32,
    parent_mask_reference: u32,
) -> Result<(), Box<dyn Error>> {
    let slot = *texture_slots
        .get(&sprite.texture_resource_index)
        .ok_or_else(|| {
            IoError::new(
                ErrorKind::InvalidData,
                "texture resource index was not loaded",
            )
        })?;
    let vertices = sprite_vertices(sprite);
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("eluna dynamic sprite vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    draws.push(GpuDraw {
        texture_slot: slot,
        vertex_buffer,
        vertex_count: vertices.len() as u32,
        pass,
        mask_reference,
        parent_mask_reference,
    });
    Ok(())
}

fn build_preview_draws(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_layout: &wgpu::BindGroupLayout,
) -> Result<(Vec<GpuTexture>, Vec<GpuDraw>, EmoteSceneBounds), Box<dyn Error>> {
    let texture = create_texture_bind_group(
        device,
        queue,
        texture_layout,
        &[255, 255, 255, 255],
        1,
        1,
        "white",
    );
    let vertices = preview_vertices()?;
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("eluna preview vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    Ok((
        vec![texture],
        vec![GpuDraw {
            texture_slot: 0,
            vertex_buffer,
            vertex_count: vertices.len() as u32,
            pass: EmoteDrawPass::Normal,
            mask_reference: 0,
            parent_mask_reference: 0,
        }],
        EmoteSceneBounds {
            min_x: -1.0,
            min_y: -1.0,
            max_x: 1.0,
            max_y: 1.0,
        },
    ))
}

fn sprite_vertices(sprite: &EmoteStaticSprite) -> Vec<GpuSpriteVertex> {
    if let Some(mesh) = &sprite.mesh {
        return mesh_sprite_vertices(sprite, mesh);
    }

    let left = sprite.left();
    let right = sprite.right();
    let top = sprite.top();
    let bottom = sprite.bottom();
    let color = [1.0, 1.0, 1.0, sprite.opacity];

    let make = |position: [f32; 2], texcoord: [f32; 2]| GpuSpriteVertex {
        position: transform_sprite_point(sprite, position),
        texcoord,
        color,
    };

    vec![
        make([left, top], [sprite.uv_left, sprite.uv_top]),
        make([left, bottom], [sprite.uv_left, sprite.uv_bottom]),
        make([right, top], [sprite.uv_right, sprite.uv_top]),
        make([right, top], [sprite.uv_right, sprite.uv_top]),
        make([left, bottom], [sprite.uv_left, sprite.uv_bottom]),
        make([right, bottom], [sprite.uv_right, sprite.uv_bottom]),
    ]
}

fn transform_sprite_point(sprite: &EmoteStaticSprite, point: [f32; 2]) -> [f32; 2] {
    let sx = if sprite.scale_x.is_finite() {
        sprite.scale_x
    } else {
        1.0
    };
    let sy = if sprite.scale_y.is_finite() {
        sprite.scale_y
    } else {
        1.0
    };
    let angle = if sprite.rotation_degrees.is_finite() {
        sprite.rotation_degrees.to_radians()
    } else {
        0.0
    };
    let cos = angle.cos();
    let sin = angle.sin();
    let dx = (point[0] - sprite.center_x) * sx;
    let dy = (point[1] - sprite.center_y) * sy;
    let local = [
        sprite.center_x + dx * cos - dy * sin,
        sprite.center_y + dx * sin + dy * cos,
    ];
    let m = sprite.world_transform;
    [
        m[0] * local[0] + m[1] * local[1] + m[4],
        m[2] * local[0] + m[3] * local[1] + m[5],
    ]
}

fn mesh_sprite_vertices(
    sprite: &EmoteStaticSprite,
    mesh: &eluna::EmoteMeshPatch,
) -> Vec<GpuSpriteVertex> {
    let division_x = mesh.division_x.max(1) as usize;
    let division_y = mesh.division_y.max(1) as usize;
    let left = sprite.left();
    let top = sprite.top();
    let color = [1.0, 1.0, 1.0, sprite.opacity];

    let vertex_at = |ix: usize, iy: usize| -> GpuSpriteVertex {
        let u = ix as f32 / division_x as f32;
        let v = iy as f32 / division_y as f32;
        let p = mesh.sample(u, v);
        let position = [left + p[0] * sprite.width, top + p[1] * sprite.height];
        GpuSpriteVertex {
            position: transform_sprite_point(sprite, position),
            texcoord: [
                sprite.uv_left + (sprite.uv_right - sprite.uv_left) * u,
                sprite.uv_top + (sprite.uv_bottom - sprite.uv_top) * v,
            ],
            color,
        }
    };

    let mut vertices = Vec::with_capacity(division_x * division_y * 6);
    for y in 0..division_y {
        for x in 0..division_x {
            let tl = vertex_at(x, y);
            let bl = vertex_at(x, y + 1);
            let tr = vertex_at(x + 1, y);
            let br = vertex_at(x + 1, y + 1);
            vertices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }
    vertices
}

/// Runs the same wgpu pipeline the live window uses, but renders one frame
/// to an offscreen Rgba8UnormSrgb texture and reads back to PNG.  Lets us
/// see the exact wgpu output without needing a window.  The offscreen target
/// uses the same format the swapchain typically uses on macOS / Win10 so
/// any sRGB / blend-mode behaviour matches the live render.
fn dump_gpu_frame(model: &LoadedModel, output_dir: &PathBuf) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output_dir)?;
    let width: u32 = 900;
    let height: u32 = 1400;
    let target_format = wgpu::TextureFormat::Rgba8UnormSrgb;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("eluna offscreen device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }))?;

    // Bind-group layouts mirror the live renderer exactly.
    let transform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("eluna offscreen transform bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("eluna offscreen texture bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let mask_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("eluna offscreen mask bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("eluna offscreen sprite shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let masked_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("eluna offscreen masked sprite shader"),
        source: wgpu::ShaderSource::Wgsl(MASKED_SHADER.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("eluna offscreen sprite pipeline layout"),
        bind_group_layouts: &[Some(&transform_bgl), Some(&texture_bgl)],
        immediate_size: 0,
    });
    let masked_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("eluna offscreen masked sprite pipeline layout"),
        bind_group_layouts: &[Some(&transform_bgl), Some(&texture_bgl), Some(&mask_bgl)],
        immediate_size: 0,
    });

    let mask_format = wgpu::TextureFormat::Rgba8Unorm;
    let pipeline = create_color_pipeline(
        &device,
        &pipeline_layout,
        &shader,
        target_format,
        "offscreen sprite",
    );
    let mask_pipeline = create_color_pipeline(
        &device,
        &pipeline_layout,
        &shader,
        mask_format,
        "offscreen mask",
    );
    let masked_pipeline = create_color_pipeline(
        &device,
        &masked_pipeline_layout,
        &masked_shader,
        target_format,
        "offscreen masked",
    );
    let mask_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("offscreen mask sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    // Build atlases and draws using the same logic as the live renderer.
    let (textures, model_texture_slots, draws, bounds) = build_model_assets(
        &device,
        &queue,
        &texture_bgl,
        &model.normalized_data,
        &model.psb,
        &model.schema,
        &model.scene,
    )?;
    eprintln!(
        "offscreen build: textures={} draws={} bounds={:?}",
        textures.len(),
        draws.len(),
        bounds
    );

    // Transform uniform: identical math to GpuState::update_transform.
    let center = bounds.center();
    let bounds_w = bounds.width().max(1.0);
    let bounds_h = bounds.height().max(1.0);
    let viewport_w = width as f32;
    let viewport_h = height as f32;
    let fit = 0.9 * (viewport_w / bounds_w).min(viewport_h / bounds_h);
    let transform = TransformUniform {
        center,
        viewport_scale: [fit * 2.0 / viewport_w, fit * 2.0 / viewport_h],
        player_coord: [0.0, 0.0],
        player_scale: 1.0,
        player_cos: 1.0,
        player_sin: 0.0,
        _pad: 0.0,
    };
    let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offscreen transform uniform"),
        contents: bytemuck::bytes_of(&transform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("offscreen transform bind group"),
        layout: &transform_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: transform_buffer.as_entire_binding(),
        }],
    });

    // Offscreen color target.
    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("eluna offscreen color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: target_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Build per-reference mask textures.
    use std::collections::BTreeSet;
    let needed_refs: BTreeSet<u32> = draws
        .iter()
        .filter(|d| d.parent_mask_reference != 0)
        .map(|d| d.parent_mask_reference)
        .collect();
    let mut mask_targets = BTreeMap::<u32, MaskCacheEntry>::new();
    for reference in needed_refs.iter().copied() {
        let label = format!("offscreen mask #{reference}");
        let (texture, view) = create_alpha_mask_target(&device, width, height, mask_format, &label);
        let bind_group = create_alpha_mask_bind_group(
            &device,
            &mask_bgl,
            &view,
            &mask_sampler,
            &format!("offscreen mask bg #{reference}"),
        );
        mask_targets.insert(
            reference,
            MaskCacheEntry {
                _texture: texture,
                view,
                bind_group,
            },
        );
    }
    let _ = model_texture_slots;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("eluna offscreen encoder"),
    });

    for reference in needed_refs.iter().copied() {
        let entry = mask_targets.get(&reference).unwrap();
        let mut mp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("offscreen mask pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &entry.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        mp.set_pipeline(&mask_pipeline);
        mp.set_bind_group(0, &transform_bind_group, &[]);
        for draw in &draws {
            if draw.mask_reference != reference {
                continue;
            }
            if draw.pass != EmoteDrawPass::MaskGeneration
                && draw.pass != EmoteDrawPass::StencilCompositeMask
            {
                continue;
            }
            let Some(t) = textures.get(draw.texture_slot) else {
                continue;
            };
            mp.set_bind_group(1, &t.bind_group, &[]);
            mp.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
            mp.draw(0..draw.vertex_count, 0..1);
        }
    }

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("offscreen color pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.05,
                        g: 0.05,
                        b: 0.06,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        #[derive(Copy, Clone, PartialEq, Eq)]
        enum Mode {
            None,
            Color,
            Masked,
        }
        let mut mode = Mode::None;
        for draw in &draws {
            if draw.pass == EmoteDrawPass::MaskGeneration
                || draw.pass == EmoteDrawPass::StencilCompositeMask
            {
                continue;
            }
            let Some(t) = textures.get(draw.texture_slot) else {
                continue;
            };
            if draw.parent_mask_reference != 0 {
                let Some(m) = mask_targets.get(&draw.parent_mask_reference) else {
                    continue;
                };
                if mode != Mode::Masked {
                    pass.set_pipeline(&masked_pipeline);
                    pass.set_bind_group(0, &transform_bind_group, &[]);
                    mode = Mode::Masked;
                }
                pass.set_bind_group(2, &m.bind_group, &[]);
                pass.set_bind_group(1, &t.bind_group, &[]);
            } else {
                if mode != Mode::Color {
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &transform_bind_group, &[]);
                    mode = Mode::Color;
                }
                pass.set_bind_group(1, &t.bind_group, &[]);
            }
            pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
            pass.draw(0..draw.vertex_count, 0..1);
        }
    }

    // Copy to readback buffer.  Rgba8 is 4 bytes/pixel; bytes_per_row must
    // be a multiple of 256 per wgpu's copy-buffer alignment rules.
    let row_bytes_aligned = ((4 * width + 255) / 256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen readback"),
        size: (row_bytes_aligned * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &color_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes_aligned),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver.recv().map_err(|e| {
        Box::new(IoError::new(ErrorKind::Other, format!("map_async: {e}"))) as Box<dyn Error>
    })??;

    let data = slice.get_mapped_range();
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let start = (row * row_bytes_aligned) as usize;
        let end = start + (width * 4) as usize;
        rgba.extend_from_slice(&data[start..end]);
    }
    drop(data);
    buffer.unmap();

    let path = output_dir.join("gpu_frame_000.png");
    let image = image::RgbaImage::from_raw(width, height, rgba)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "rgba buffer size mismatch"))?;
    image.save(&path)?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

fn dump_runtime_frames(
    model: &mut LoadedModel,
    output_dir: &PathBuf,
    frame_count: usize,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output_dir)?;
    let textures = decode_model_textures(&model.normalized_data, &model.psb, &model.schema)?;
    let motion_name = model.active_motion.clone().ok_or_else(|| {
        IoError::new(
            ErrorKind::InvalidInput,
            "frame dump requires an active motion",
        )
    })?;

    for index in 0..frame_count {
        let previous_variables = player_variable_values(&model.player);
        let previous_scene = model.scene.clone();
        let delta_ticks = if index == 0 { 0.0 } else { 4.0 };
        if delta_ticks > 0.0 {
            model.player.progress_ticks_without_physics(delta_ticks);
        }
        rebuild_loaded_model_scene(model, delta_ticks)?;
        let variables = player_variable_values(&model.player);
        let scene = model.scene.clone();
        let path = output_dir.join(format!("frame_{index:03}.png"));
        write_scene_png(&scene, &textures, &path, 900, 1400)?;
        if model.debug_frames {
            debug_runtime_frame(
                model,
                &motion_name,
                delta_ticks,
                &previous_variables,
                &previous_scene,
                &scene,
                &variables,
            );
        }
    }

    eprintln!(
        "exported {frame_count} runtime frames to {}",
        output_dir.display()
    );
    Ok(())
}

fn decode_model_textures(
    data: &[u8],
    psb: &PsbFile,
    schema: &EmoteModelSchema,
) -> Result<BTreeMap<u32, DecodedTexture>, Box<dyn Error>> {
    let mut textures = BTreeMap::new();
    for texture_source in schema.textures.values() {
        if textures.contains_key(&texture_source.resource_index) {
            continue;
        }
        let bytes = psb
            .resource_bytes(data, texture_source.resource_index as usize)
            .ok_or_else(|| {
                IoError::new(
                    ErrorKind::InvalidData,
                    "texture resource index out of range",
                )
            })?;
        let decoded = decode_texture_rgba(
            bytes,
            texture_source.width,
            texture_source.height,
            texture_source.format.as_deref(),
            texture_source.compress.as_deref(),
            texture_source.bit_count,
            schema.spec.as_deref(),
        )?;
        textures.insert(texture_source.resource_index, decoded);
    }
    Ok(textures)
}

fn write_scene_png(
    scene: &EmoteStaticScene,
    textures: &BTreeMap<u32, DecodedTexture>,
    path: &PathBuf,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn Error>> {
    let bounds = scene.bounds.unwrap_or(EmoteSceneBounds {
        min_x: -450.0,
        min_y: -700.0,
        max_x: 450.0,
        max_y: 1100.0,
    });
    let mut rgba = vec![15u8; width as usize * height as usize * 4];
    for px in rgba.chunks_exact_mut(4) {
        px[0] = 13;
        px[1] = 13;
        px[2] = 16;
        px[3] = 255;
    }

    for sprite in scene
        .sprites
        .iter()
        .filter(|sprite| sprite.visible && sprite.opacity > 0.0)
    {
        if sprite.draw_frame_info.pass == EmoteDrawPass::MaskGeneration {
            continue;
        }
        let Some(texture) = textures.get(&sprite.texture_resource_index) else {
            continue;
        };
        let vertices = sprite_vertices(sprite);
        for tri in vertices.chunks_exact(3) {
            raster_triangle(&mut rgba, width, height, bounds, texture, tri);
        }
    }

    let image = image::RgbaImage::from_raw(width, height, rgba)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid software frame dimensions"))?;
    image.save(path)?;
    Ok(())
}

fn raster_triangle(
    target: &mut [u8],
    width: u32,
    height: u32,
    bounds: EmoteSceneBounds,
    texture: &DecodedTexture,
    tri: &[GpuSpriteVertex],
) {
    let to_screen = |position: [f32; 2]| -> [f32; 2] {
        let sx = (position[0] - bounds.min_x) / (bounds.max_x - bounds.min_x).max(f32::EPSILON);
        let sy = (position[1] - bounds.min_y) / (bounds.max_y - bounds.min_y).max(f32::EPSILON);
        [sx * (width as f32 - 1.0), sy * (height as f32 - 1.0)]
    };
    let p0 = to_screen(tri[0].position);
    let p1 = to_screen(tri[1].position);
    let p2 = to_screen(tri[2].position);
    let min_x = p0[0].min(p1[0]).min(p2[0]).floor().max(0.0) as u32;
    let max_x = p0[0].max(p1[0]).max(p2[0]).ceil().min(width as f32 - 1.0) as u32;
    let min_y = p0[1].min(p1[1]).min(p2[1]).floor().max(0.0) as u32;
    let max_y = p0[1].max(p1[1]).max(p2[1]).ceil().min(height as f32 - 1.0) as u32;
    let area = edge(p0, p1, p2);
    if area.abs() <= f32::EPSILON {
        return;
    }

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = [x as f32 + 0.5, y as f32 + 0.5];
            let w0 = edge(p1, p2, p) / area;
            let w1 = edge(p2, p0, p) / area;
            let w2 = edge(p0, p1, p) / area;
            if w0 < -0.0001 || w1 < -0.0001 || w2 < -0.0001 {
                continue;
            }
            let u = tri[0].texcoord[0] * w0 + tri[1].texcoord[0] * w1 + tri[2].texcoord[0] * w2;
            let v = tri[0].texcoord[1] * w0 + tri[1].texcoord[1] * w1 + tri[2].texcoord[1] * w2;
            let color_alpha = tri[0].color[3] * w0 + tri[1].color[3] * w1 + tri[2].color[3] * w2;
            let tx = (u.clamp(0.0, 1.0) * (texture.width as f32 - 1.0)).round() as u32;
            let ty = (v.clamp(0.0, 1.0) * (texture.height as f32 - 1.0)).round() as u32;
            let src_index = (ty as usize * texture.width as usize + tx as usize) * 4;
            let dst_index = (y as usize * width as usize + x as usize) * 4;
            let src_a = (texture.rgba[src_index + 3] as f32 / 255.0) * color_alpha;
            if src_a <= 0.0 {
                continue;
            }
            for channel in 0..3 {
                let src = texture.rgba[src_index + channel] as f32 / 255.0;
                let dst = target[dst_index + channel] as f32 / 255.0;
                target[dst_index + channel] = ((src * src_a + dst * (1.0 - src_a)) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
            target[dst_index + 3] = 255;
        }
    }
}

fn edge(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
}

#[derive(Clone)]
struct DecodedTexture {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
}

fn decode_texture_rgba(
    bytes: &[u8],
    expected_width: u32,
    expected_height: u32,
    format: Option<&str>,
    compress: Option<&str>,
    bit_count: Option<u32>,
    spec: Option<&str>,
) -> Result<DecodedTexture, Box<dyn Error>> {
    let bpp = bit_count.unwrap_or(32);
    let expected_len =
        expected_width as usize * expected_height as usize * (bpp as usize / 8).max(1);

    let payload = if is_rl_compressed(compress) {
        decompress_rl(bytes, (bpp as usize / 8).max(1), expected_len)?
    } else {
        bytes.to_vec()
    };

    if bpp == 32 && payload.len() == expected_width as usize * expected_height as usize * 4 {
        let rgba = convert_raw_32bpp_texture_to_rgba(&payload, format, spec);
        return Ok(DecodedTexture {
            rgba,
            width: expected_width,
            height: expected_height,
        });
    }

    if !is_rl_compressed(compress) {
        if let Ok(image) = image::load_from_memory(bytes) {
            let (width, height) = image.dimensions();
            return Ok(DecodedTexture {
                rgba: image.to_rgba8().into_raw(),
                width,
                height,
            });
        }
    }

    Err(Box::new(IoError::new(
        ErrorKind::InvalidData,
        format!(
            "unsupported texture payload: format={:?} compress={:?} bit_count={:?} spec={:?} bytes={} expected={}",
            format, compress, bit_count, spec, bytes.len(), expected_len
        ),
    )))
}

fn convert_raw_32bpp_texture_to_rgba(
    bytes: &[u8],
    format: Option<&str>,
    spec: Option<&str>,
) -> Vec<u8> {
    let order = raw_32bpp_channel_order(format, spec);
    let mut rgba = Vec::with_capacity(bytes.len());

    for px in bytes.chunks_exact(4) {
        match order {
            Raw32ChannelOrder::Bgra => rgba.extend_from_slice(&[px[2], px[1], px[0], px[3]]),
            Raw32ChannelOrder::Rgba => rgba.extend_from_slice(px),
            Raw32ChannelOrder::Bgrx => rgba.extend_from_slice(&[px[2], px[1], px[0], 0xff]),
            Raw32ChannelOrder::Rgbx => rgba.extend_from_slice(&[px[0], px[1], px[2], 0xff]),
        }
    }

    rgba
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Raw32ChannelOrder {
    Bgra,
    Rgba,
    Bgrx,
    Rgbx,
}

fn raw_32bpp_channel_order(format: Option<&str>, spec: Option<&str>) -> Raw32ChannelOrder {
    let Some(format) = format else {
        return if spec_uses_big_endian_rgba(spec) {
            Raw32ChannelOrder::Rgba
        } else {
            Raw32ChannelOrder::Bgra
        };
    };

    let normalized: String = format
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    match normalized.as_str() {
        // `RGBA8` is spec-dependent in FreeMote: win/krkr are LeRGBA8
        // (BGRA bytes in a Windows DIB), while common/ems/web-style data is
        // BeRGBA8 (RGBA bytes). Tyrano's emtbytes samples are normally common
        // or ems, so using the root `spec` is required to avoid red/blue swap.
        "RGBA" | "RGBA8" => {
            if spec_uses_big_endian_rgba(spec) {
                Raw32ChannelOrder::Rgba
            } else {
                Raw32ChannelOrder::Bgra
            }
        }
        "BERGBA8" => Raw32ChannelOrder::Rgba,
        "LERGBA8" | "BGRA8" | "ARGB8" | "A8R8G8B8" | "D3DFMTA8R8G8B8" => Raw32ChannelOrder::Bgra,
        "BGRX8" | "X8R8G8B8" | "D3DFMTX8R8G8B8" => Raw32ChannelOrder::Bgrx,
        "RGBX8" | "RGBX" => Raw32ChannelOrder::Rgbx,
        _ => {
            if spec_uses_big_endian_rgba(spec) {
                Raw32ChannelOrder::Rgba
            } else {
                Raw32ChannelOrder::Bgra
            }
        }
    }
}

fn spec_uses_big_endian_rgba(spec: Option<&str>) -> bool {
    let Some(spec) = spec else {
        return false;
    };
    matches!(
        spec.to_ascii_lowercase().as_str(),
        "common" | "ems" | "vita" | "psp" | "ps3"
    )
}

fn is_rl_compressed(compress: Option<&str>) -> bool {
    compress.is_some_and(|value| value.eq_ignore_ascii_case("RL"))
}

fn decompress_rl(
    bytes: &[u8],
    align: usize,
    expected_len: usize,
) -> Result<Vec<u8>, Box<dyn Error>> {
    if align == 0 {
        return Err(Box::new(IoError::new(
            ErrorKind::InvalidData,
            "invalid RL align",
        )));
    }
    let mut out = Vec::with_capacity(expected_len);
    let mut pos = 0usize;
    while pos < bytes.len() {
        let cmd = bytes[pos];
        pos += 1;
        if (cmd & 0x80) != 0 {
            let count = ((cmd ^ 0x80) as usize) + 3;
            let end = pos
                .checked_add(align)
                .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "RL overflow"))?;
            let pattern = bytes.get(pos..end).ok_or_else(|| {
                IoError::new(ErrorKind::UnexpectedEof, "truncated RL repeat block")
            })?;
            for _ in 0..count {
                out.extend_from_slice(pattern);
            }
            pos = end;
        } else {
            let count = ((cmd as usize) + 1) * align;
            let end = pos
                .checked_add(count)
                .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "RL overflow"))?;
            let block = bytes.get(pos..end).ok_or_else(|| {
                IoError::new(ErrorKind::UnexpectedEof, "truncated RL literal block")
            })?;
            out.extend_from_slice(block);
            pos = end;
        }
        if out.len() > expected_len {
            return Err(Box::new(IoError::new(
                ErrorKind::InvalidData,
                "RL output exceeds expected length",
            )));
        }
    }
    if out.len() != expected_len {
        return Err(Box::new(IoError::new(
            ErrorKind::InvalidData,
            format!(
                "RL output length mismatch: got {}, expected {}",
                out.len(),
                expected_len
            ),
        )));
    }
    Ok(out)
}

fn create_texture_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    rgba: &[u8],
    width: u32,
    height: u32,
    label: &str,
) -> GpuTexture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("eluna sprite sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("eluna sprite texture bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    GpuTexture {
        _texture: texture,
        _view: view,
        _sampler: sampler,
        bind_group,
    }
}

// ── egui UI ──────────────────────────────────────────────────────────────────

fn build_ui(ctx: &egui::Context, model: Option<&mut LoadedModel>, state: &mut UiState) {
    egui::TopBottomPanel::top("eluna_toolbar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.strong("Eluna");
            ui.separator();
            ui.toggle_value(&mut state.show_motion, "Motion");
            ui.toggle_value(&mut state.show_variables, "Variables");
            ui.toggle_value(&mut state.show_selector, "Selector");
            ui.toggle_value(&mut state.show_face, "Face");
            ui.toggle_value(&mut state.show_physics, "Physics");
            ui.toggle_value(&mut state.show_layers, "Layers");
            ui.toggle_value(&mut state.show_textures, "Textures");
            ui.toggle_value(&mut state.show_api_log, "API Log");
        });
    });

    let Some(model) = model else { return };

    if state.show_motion {
        let mut open = state.show_motion;
        egui::Window::new("Motion / Timeline")
            .open(&mut open)
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                panel_motion(ui, model, state);
            });
        state.show_motion = open;
    }

    if state.show_variables {
        let mut open = state.show_variables;
        egui::Window::new("Variables")
            .open(&mut open)
            .resizable(true)
            .default_width(360.0)
            .default_height(500.0)
            .show(ctx, |ui| {
                panel_variables(ui, model, state);
            });
        state.show_variables = open;
    }

    if state.show_selector {
        let mut open = state.show_selector;
        egui::Window::new("Selector / Parts")
            .open(&mut open)
            .resizable(true)
            .default_width(300.0)
            .show(ctx, |ui| {
                panel_selector(ui, model, state);
            });
        state.show_selector = open;
    }

    if state.show_face {
        let mut open = state.show_face;
        egui::Window::new("Face / Expression")
            .open(&mut open)
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                panel_face(ui, model, state);
            });
        state.show_face = open;
    }

    if state.show_physics {
        let mut open = state.show_physics;
        egui::Window::new("Physics")
            .open(&mut open)
            .resizable(true)
            .default_width(340.0)
            .show(ctx, |ui| {
                panel_physics(ui, model, state);
            });
        state.show_physics = open;
    }

    if state.show_layers {
        let mut open = state.show_layers;
        egui::Window::new("Layers / DrawFrameInfo")
            .open(&mut open)
            .resizable(true)
            .default_width(620.0)
            .default_height(400.0)
            .show(ctx, |ui| {
                panel_layers(ui, model, state);
            });
        state.show_layers = open;
    }

    if state.show_textures {
        let mut open = state.show_textures;
        egui::Window::new("Textures / Atlas")
            .open(&mut open)
            .resizable(true)
            .default_width(340.0)
            .show(ctx, |ui| {
                panel_textures(ui, model, state);
            });
        state.show_textures = open;
    }

    if state.show_api_log {
        let mut open = state.show_api_log;
        egui::Window::new("Official API Log")
            .open(&mut open)
            .resizable(true)
            .default_width(520.0)
            .default_height(360.0)
            .show(ctx, |ui| {
                panel_api_log(ui, model, state);
            });
        state.show_api_log = open;
    }
}

fn panel_motion(ui: &mut egui::Ui, model: &mut LoadedModel, state: &mut UiState) {
    ui.horizontal(|ui| {
        ui.label("Motion:");
        let current = model.active_motion.clone().unwrap_or_default();
        egui::ComboBox::from_id_salt("motion_select")
            .selected_text(if current.is_empty() {
                "(none)"
            } else {
                &current
            })
            .show_ui(ui, |ui| {
                if let Ok(infos) = model.schema.motion_infos(&model.psb) {
                    for info in infos {
                        let selected = model.active_motion.as_deref() == Some(&info.name);
                        if ui.selectable_label(selected, &info.name).clicked() {
                            model.active_motion = Some(info.name.clone());
                            state.dirty = true;
                        }
                    }
                }
            });
    });

    ui.separator();
    ui.label("Official player model: main timeline + 6 difference timeline slots");
    let render_defaults = EmoteDeviceRenderOptions::default();
    ui.label(format!(
        "Render defaults: mask={:?}, protectTranslucentTextureColor={}, maskRegionClipping={}",
        render_defaults.mask_mode,
        render_defaults.protect_translucent_texture_color,
        render_defaults.mask_region_clipping
    ));

    let main_names: Vec<String> = model
        .player
        .main_timeline_labels()
        .into_iter()
        .map(str::to_owned)
        .collect();
    let diff_names: Vec<String> = model
        .player
        .diff_timeline_labels()
        .into_iter()
        .map(str::to_owned)
        .collect();
    ui.label(format!(
        "official labels: main={} diff={} playing={}",
        main_names.len(),
        diff_names.len(),
        model.player.playing_timeline_info().len()
    ));

    if state.main_timeline.is_empty() {
        if let Some((name, _)) = model
            .player
            .active_timelines()
            .iter()
            .find(|(name, mode)| !name.starts_with("@control/") && !mode.is_difference())
        {
            state.main_timeline = name.clone();
        }
    }

    ui.horizontal(|ui| {
        ui.label("Main timeline:");
        let selected = if state.main_timeline.is_empty() {
            "(none)"
        } else {
            state.main_timeline.as_str()
        };
        egui::ComboBox::from_id_salt("main_timeline_select")
            .selected_text(selected)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(state.main_timeline.is_empty(), "(none)")
                    .clicked()
                {
                    if !state.main_timeline.is_empty() {
                        model.player.stop_timeline(&state.main_timeline);
                    }
                    state.main_timeline.clear();
                    state.dirty = true;
                }
                for name in &main_names {
                    let selected = state.main_timeline == *name;
                    if ui.selectable_label(selected, name).clicked() {
                        if !state.main_timeline.is_empty() && state.main_timeline != *name {
                            model.player.stop_timeline(&state.main_timeline);
                        }
                        state.main_timeline = name.clone();
                        model.player.play_timeline(
                            name,
                            TimelinePlayMode::PARALLEL.with_looping(state.timeline_loop),
                        );
                        state.dirty = true;
                    }
                }
            });
    });

    ui.horizontal(|ui| {
        if ui
            .checkbox(&mut state.timeline_loop, "Loop main locally")
            .changed()
        {
            if !state.main_timeline.is_empty() {
                let elapsed = model.player.timeline_elapsed_ticks(&state.main_timeline);
                model.player.play_timeline(
                    &state.main_timeline,
                    TimelinePlayMode::PARALLEL.with_looping(state.timeline_loop),
                );
                let _ = model
                    .player
                    .set_timeline_time(&state.main_timeline, elapsed);
                state.dirty = true;
            }
        }
        ui.label("Speed:");
        if ui
            .add(egui::Slider::new(&mut state.playback_speed, 0.0_f32..=4.0).step_by(0.05))
            .changed()
        {
            state.dirty = true;
        }
        if ui.checkbox(&mut state.step_update, "Step update").changed() {
            state.dirty = true;
        }
    });

    ui.horizontal(|ui| {
        let paused = model.player.is_paused();
        if ui.button(if paused { "Play" } else { "Pause" }).clicked() {
            model.player.set_paused(!paused);
            state.dirty = true;
        }
        if ui.button("Stop all timelines").clicked() {
            model.player.stop_timeline("");
            state.main_timeline.clear();
            state.diff_timeline_slots = Default::default();
            state.dirty = true;
        }
        if ui.button("Reset active timelines").clicked() {
            let active: Vec<(String, TimelinePlayMode)> = model
                .player
                .active_timelines()
                .iter()
                .filter(|(name, _)| !name.starts_with("@control/"))
                .map(|(name, mode)| (name.clone(), *mode))
                .collect();
            for (name, mode) in active {
                model.player.stop_timeline(&name);
                model.player.play_timeline(&name, mode);
            }
            state.dirty = true;
        }
    });

    ui.horizontal(|ui| {
        ui.label("Diff fadeout ms:");
        ui.add(
            egui::DragValue::new(&mut state.diff_fadeout_ms)
                .speed(10.0)
                .range(0.0..=5000.0),
        );
    });

    for slot in 0..6 {
        ui.horizontal(|ui| {
            ui.label(format!("Diff slot {}:", slot + 1));
            let current = if state.diff_timeline_slots[slot].is_empty() {
                "(none)".to_owned()
            } else {
                state.diff_timeline_slots[slot].clone()
            };
            egui::ComboBox::from_id_salt(format!("diff_slot_{slot}"))
                .selected_text(current)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(state.diff_timeline_slots[slot].is_empty(), "(none)")
                        .clicked()
                    {
                        let prev = std::mem::take(&mut state.diff_timeline_slots[slot]);
                        if !prev.is_empty() {
                            model.player.fade_out_timeline(
                                &prev,
                                milliseconds_to_emote_ticks(state.diff_fadeout_ms),
                                0.0,
                            );
                        }
                        state.dirty = true;
                    }
                    for name in &diff_names {
                        let selected = state.diff_timeline_slots[slot] == *name;
                        if ui.selectable_label(selected, name).clicked() {
                            let prev = std::mem::replace(
                                &mut state.diff_timeline_slots[slot],
                                name.clone(),
                            );
                            if !prev.is_empty() && prev != *name {
                                model.player.fade_out_timeline(
                                    &prev,
                                    milliseconds_to_emote_ticks(state.diff_fadeout_ms),
                                    0.0,
                                );
                            }
                            model
                                .player
                                .play_timeline(name, TimelinePlayMode::PARALLEL_DIFFERENCE);
                            state.dirty = true;
                        }
                    }
                });
        });
    }

    ui.separator();
    ui.label("Active timeline scrub:");
    let active: Vec<String> = model
        .player
        .active_timelines()
        .keys()
        .filter(|name| !name.starts_with("@control/"))
        .cloned()
        .collect();
    for name in &active {
        if let Some(tl) = model.player.timelines().get(name.as_str()) {
            let dur = tl.duration_ticks;
            let mut elapsed = model.player.timeline_elapsed_ticks(name);
            if ui
                .add(egui::Slider::new(&mut elapsed, 0.0..=dur.max(0.0)).text(name))
                .changed()
            {
                if let Err(err) = model.player.set_timeline_time(name, elapsed) {
                    ui.colored_label(egui::Color32::YELLOW, err);
                }
                state.dirty = true;
            }
            ui.label(format!(
                "{}: {:.1} / {:.1} ticks ({:.0} / {:.0} ms)",
                name,
                elapsed,
                dur,
                emote_ticks_to_milliseconds(elapsed),
                emote_ticks_to_milliseconds(dur)
            ));
        }
    }

    ui.separator();
    ui.label("Transform / physics order mask");
    egui::ComboBox::from_id_salt("transform_order_mask")
        .selected_text(format!("0x{:03x}", state.transform_order_mask))
        .show_ui(ui, |ui| {
            for (label, value) in [
                ("default/mix 0x201", transform_order_mask::TYRANO_MIX),
                ("orthogonal 0x101", transform_order_mask::TYRANO_ORTHOGONAL),
                (
                    "perspective 0x202",
                    transform_order_mask::TYRANO_PERSPECTIVE,
                ),
                ("SDK default", transform_order_mask::DEFAULT),
            ] {
                if ui
                    .selectable_label(state.transform_order_mask == value, label)
                    .clicked()
                {
                    state.transform_order_mask = value;
                    model.player.set_transform_order_mask(value);
                    state.dirty = true;
                }
            }
        });

    ui.separator();
    ui.label(format!(
        "Elapsed: {:.1} ticks",
        model.player.elapsed_ticks()
    ));
}

fn panel_variables(ui: &mut egui::Ui, model: &mut LoadedModel, state: &mut UiState) {
    ui.horizontal(|ui| {
        ui.label("Filter:");
        ui.text_edit_singleline(&mut state.var_filter);
        if ui.small_button("x").clicked() {
            state.var_filter.clear();
        }
    });

    if ui.button("Reset All to Default").clicked() {
        model.player.reset_all_variables_to_default();
        state.dirty = true;
    }

    ui.separator();

    let filter = state.var_filter.to_lowercase();
    let var_infos: Vec<(String, f32, Option<f32>, Option<f32>)> = model
        .player
        .variables()
        .iter()
        .filter(|(name, _)| filter.is_empty() || name.to_lowercase().contains(&filter))
        .map(|(name, vs)| (name.clone(), vs.value, vs.info.min_value, vs.info.max_value))
        .collect();

    egui::ScrollArea::vertical().show(ui, |ui| {
        for (name, value, min_val, max_val) in var_infos {
            let frames = model
                .player
                .variables()
                .get(&name)
                .map(|state| state.info.frames.clone())
                .unwrap_or_default();
            let frame_min = frames.iter().map(|frame| frame.value).reduce(f32::min);
            let frame_max = frames.iter().map(|frame| frame.value).reduce(f32::max);
            let verified_min = min_val.or(frame_min);
            let verified_max = max_val.or(frame_max);
            let (min, max, range_note) = match (verified_min, verified_max) {
                (Some(min), Some(max)) if min < max => (min, max, None),
                (Some(value), Some(_)) | (Some(value), None) | (None, Some(value)) => (
                    value - 1.0,
                    value + 1.0,
                    Some("single-key range: debug +/-1"),
                ),
                (None, None) => (
                    value - 1.0,
                    value + 1.0,
                    Some("unknown range: debug current +/-1"),
                ),
            };
            ui.horizontal(|ui| {
                let mut v = value;
                if ui
                    .add(
                        egui::Slider::new(&mut v, min..=max)
                            .text(&name)
                            .drag_value_speed(0.05),
                    )
                    .changed()
                {
                    model.player.set_variable_immediate(&name, v);
                    state.dirty = true;
                }
                if !frames.is_empty() {
                    egui::ComboBox::from_id_salt(format!("var_frame_{name}"))
                        .selected_text("frame")
                        .show_ui(ui, |ui| {
                            for frame in &frames {
                                let label = if frame.label.is_empty() {
                                    format!("{:.3}", frame.value)
                                } else {
                                    format!("{} = {:.3}", frame.label, frame.value)
                                };
                                if ui.selectable_label(false, label).clicked() {
                                    model.player.set_variable_immediate(&name, frame.value);
                                    state.dirty = true;
                                }
                            }
                        });
                }
                if let Some(note) = range_note {
                    ui.colored_label(egui::Color32::YELLOW, note);
                }
                if ui
                    .small_button("R")
                    .on_hover_text("Reset to default")
                    .clicked()
                {
                    model.player.reset_variable_to_default(&name);
                    state.dirty = true;
                }
            });
        }
    });
}

fn panel_selector(ui: &mut egui::Ui, model: &mut LoadedModel, state: &mut UiState) {
    let pipeline = model.player.runtime_pipeline().clone();

    if !pipeline.selector_controls.is_empty() {
        ui.heading("Selectors");
        for control in &pipeline.selector_controls {
            if !control.enabled {
                continue;
            }
            ui.horizontal(|ui| {
                ui.label(format!("{}:", control.label));
                let current_val = model.player.variable_value(&control.label).unwrap_or(0.0);
                let current_idx = current_val.round() as usize;
                let current_name = control
                    .option_list
                    .get(current_idx)
                    .map(|o| o.label.as_str())
                    .unwrap_or("?");
                egui::ComboBox::from_id_salt(format!("sel_{}", control.label))
                    .selected_text(current_name)
                    .show_ui(ui, |ui| {
                        for (idx, option) in control.option_list.iter().enumerate() {
                            if ui
                                .selectable_label(current_idx == idx, &option.label)
                                .clicked()
                            {
                                if let Err(err) =
                                    model.player.set_selector_option(&control.label, idx)
                                {
                                    eprintln!("selector update error: {err}");
                                }
                                state.dirty = true;
                            }
                        }
                    });
            });
            ui.indent(format!("seld_{}", control.label), |ui| {
                for (idx, option) in control.option_list.iter().enumerate() {
                    ui.label(format!(
                        "[{}] {} off={} on={}",
                        idx, option.label, option.off_value, option.on_value
                    ));
                }
            });
        }
    }

    if !pipeline.parts_controls.is_empty() {
        ui.separator();
        ui.heading("Parts Controls");
        for control in &pipeline.parts_controls {
            let label = control.label.as_deref().unwrap_or("?");
            ui.label(format!("• {} (enabled={})", label, control.enabled));
        }
    }
}

fn panel_face(ui: &mut egui::Ui, model: &mut LoadedModel, state: &mut UiState) {
    let pipeline = model.player.runtime_pipeline().clone();

    if !pipeline.eye_controls.is_empty() {
        ui.heading("Eye Controls");
        for control in &pipeline.eye_controls {
            let label = control.label.as_deref().unwrap_or("?");
            ui.label(format!("• {} enabled={}", label, control.enabled));
        }
        ui.separator();
        // Show eye-related variables dynamically from the variable map
        let eye_vars: Vec<(String, f32, Option<f32>, Option<f32>)> = model
            .player
            .variables()
            .iter()
            .filter(|(name, _)| {
                let n = name.to_lowercase();
                n.contains("eye") || n.contains("hitomi") || n.contains("目")
            })
            .map(|(name, s)| (name.clone(), s.value, s.info.min_value, s.info.max_value))
            .collect();
        for (name, value, min_val, max_val) in eye_vars {
            let min = min_val.unwrap_or(-30.0);
            let max = max_val.unwrap_or(30.0);
            let mut v = value;
            if ui
                .add(egui::Slider::new(&mut v, min..=max).text(&name))
                .changed()
            {
                model.player.set_variable_immediate(&name, v);
                state.dirty = true;
            }
        }
    }

    if !pipeline.eyebrow_controls.is_empty() {
        ui.separator();
        ui.heading("Eyebrow Controls");
        for control in &pipeline.eyebrow_controls {
            let label = control.label.as_deref().unwrap_or("?");
            ui.label(format!("• {} enabled={}", label, control.enabled));
        }
        let eyebrow_vars: Vec<(String, f32, Option<f32>, Option<f32>)> = model
            .player
            .variables()
            .iter()
            .filter(|(name, _)| {
                let n = name.to_lowercase();
                n.contains("eyebrow") || n.contains("brow") || n.contains("眉")
            })
            .map(|(name, s)| (name.clone(), s.value, s.info.min_value, s.info.max_value))
            .collect();
        for (name, value, min_val, max_val) in eyebrow_vars {
            let min = min_val.unwrap_or(-30.0);
            let max = max_val.unwrap_or(30.0);
            let mut v = value;
            if ui
                .add(egui::Slider::new(&mut v, min..=max).text(&name))
                .changed()
            {
                model.player.set_variable_immediate(&name, v);
                state.dirty = true;
            }
        }
    }

    if !pipeline.mouth_controls.is_empty() {
        ui.separator();
        ui.heading("Mouth Controls");
        for control in &pipeline.mouth_controls {
            let label = control.label.as_deref().unwrap_or("?");
            ui.label(format!("• {} enabled={}", label, control.enabled));
        }
        let mouth_vars: Vec<(String, f32, Option<f32>, Option<f32>)> = model
            .player
            .variables()
            .iter()
            .filter(|(name, _)| {
                let n = name.to_lowercase();
                n.contains("mouth") || n.contains("口")
            })
            .map(|(name, s)| (name.clone(), s.value, s.info.min_value, s.info.max_value))
            .collect();
        for (name, value, min_val, max_val) in mouth_vars {
            let min = min_val.unwrap_or(-30.0);
            let max = max_val.unwrap_or(30.0);
            let mut v = value;
            if ui
                .add(egui::Slider::new(&mut v, min..=max).text(&name))
                .changed()
            {
                model.player.set_variable_immediate(&name, v);
                state.dirty = true;
            }
        }
    }

    ui.separator();
    ui.heading("Face Layer Status");
    let face_layers: Vec<_> = model
        .scene
        .sprites
        .iter()
        .filter(|s| is_face_related_sprite(s))
        .map(|s| {
            (
                s.draw_frame_info.path.clone(),
                s.opacity,
                s.visible,
                s.texture_resource_index,
                s.draw_frame_info.pass,
            )
        })
        .collect();
    egui::ScrollArea::vertical()
        .max_height(200.0)
        .show(ui, |ui| {
            for (path, opacity, visible, tex_idx, pass) in &face_layers {
                let status = if !visible {
                    "HIDDEN"
                } else if *opacity <= 0.0 {
                    "op=0"
                } else if *pass == EmoteDrawPass::MaskGeneration {
                    "mask-gen"
                } else {
                    "ok"
                };
                let short_path = path.rsplit('/').next().unwrap_or(path.as_str());
                let color = if status == "ok" {
                    egui::Color32::LIGHT_GREEN
                } else {
                    egui::Color32::YELLOW
                };
                ui.colored_label(
                    color,
                    format!("{status} [{short_path}] op={opacity:.2} tex={tex_idx}"),
                );
            }
        });
}

fn panel_api_log(ui: &mut egui::Ui, model: &mut LoadedModel, state: &mut UiState) {
    ui.horizontal(|ui| {
        if ui.button("Start record").clicked() {
            model.player.start_record_api_log();
            state.api_log_text.clear();
        }
        if ui.button("Stop record").clicked() {
            model.player.stop_record_api_log();
            state.api_log_text = model.player.api_log();
        }
        if ui.button("Refresh text").clicked() {
            state.api_log_text = model.player.api_log();
        }
        if ui.button("Set log").clicked() {
            model.player.set_api_log(&state.api_log_text);
        }
        if ui.button("Replay once").clicked() {
            model.player.set_api_log(&state.api_log_text);
            model.player.replay_api_log_once();
            state.dirty = true;
        }
        if ui.button("Clear").clicked() {
            model.player.clear_api_log();
            state.api_log_text.clear();
        }
    });
    ui.label(format!(
        "recording={} replaying={} modified={} animating={}",
        model.player.is_recording_api_log(),
        model.player.is_replaying_api_log(),
        model.player.is_modified(),
        model.player.is_animating()
    ));
    ui.label("Format: one tab-separated official-style command per line, e.g. PlayTimeline<TAB>label<TAB>flags");
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.add(
            egui::TextEdit::multiline(&mut state.api_log_text)
                .desired_rows(16)
                .desired_width(f32::INFINITY),
        );
    });
}

fn panel_physics(ui: &mut egui::Ui, model: &mut LoadedModel, state: &mut UiState) {
    ui.horizontal(|ui| {
        let mut enabled = model.player.is_physics_enabled();
        if ui.checkbox(&mut enabled, "Physics enabled").changed() {
            model.player.set_physics_enabled(enabled);
            state.dirty = true;
        }
        if ui.button("Reset").clicked() {
            model.player.reset_physics();
            state.dirty = true;
        }
    });

    ui.separator();
    ui.heading("Official outer physics inputs");
    for label in ["bust", "parts", "hair"] {
        let mut force = model.player.outer_force(label);
        ui.horizontal(|ui| {
            ui.label(format!("{label} outer force"));
            let changed_x = ui
                .add(egui::DragValue::new(&mut force[0]).speed(0.1))
                .changed();
            let changed_y = ui
                .add(egui::DragValue::new(&mut force[1]).speed(0.1))
                .changed();
            if changed_x || changed_y {
                model
                    .player
                    .set_outer_force(label, force[0], force[1], 0.0, 0.0);
                state.dirty = true;
            }
        });
    }
    let mut outer_rot = model.player.outer_rot();
    if ui
        .add(egui::Slider::new(&mut outer_rot, -180.0..=180.0).text("outer rot"))
        .changed()
    {
        model.player.set_outer_rot(outer_rot, 0.0, 0.0);
        state.dirty = true;
    }
    let mut bust_scale = model.player.bust_scale();
    let mut hair_scale = model.player.hair_scale();
    let mut parts_scale = model.player.parts_scale();
    if ui
        .add(egui::Slider::new(&mut bust_scale, 0.0..=4.0).text("global bust scale"))
        .changed()
    {
        model.player.set_bust_scale(bust_scale);
        state.dirty = true;
    }
    if ui
        .add(egui::Slider::new(&mut hair_scale, 0.0..=4.0).text("global hair scale"))
        .changed()
    {
        model.player.set_hair_scale(hair_scale);
        state.dirty = true;
    }
    if ui
        .add(egui::Slider::new(&mut parts_scale, 0.0..=4.0).text("global parts scale"))
        .changed()
    {
        model.player.set_parts_scale(parts_scale);
        state.dirty = true;
    }

    ui.separator();

    let pipeline = model.player.runtime_pipeline().clone();
    let variables = model.player.variables().clone();

    for (idx, control) in pipeline.physics_controls.iter().enumerate() {
        let (kind, def) = match control {
            PhysicsControl::Bust(d) => ("Bust", d),
            PhysicsControl::Hair(d) => ("Hair", d),
            PhysicsControl::Parts(d) => ("Parts", d),
        };
        let header = format!(
            "{kind}: {} ({})",
            def.label,
            if def.enabled { "on" } else { "off" }
        );
        egui::CollapsingHeader::new(header)
            .id_salt(format!("phys_{idx}"))
            .show(ui, |ui| {
                ui.label(format!("baseLayer: {:?}", def.base_layer));
                if let Some(vname) = &def.var_lr {
                    let v = variables.get(vname).map(|s| s.value).unwrap_or(0.0);
                    ui.label(format!("var_lr: {vname} = {v:.3}"));
                }
                if let Some(vname) = &def.var_ud {
                    let v = variables.get(vname).map(|s| s.value).unwrap_or(0.0);
                    ui.label(format!("var_ud: {vname} = {v:.3}"));
                }
                if let Some(vname) = &def.var_lrm {
                    let v = variables.get(vname).map(|s| s.value).unwrap_or(0.0);
                    ui.label(format!("var_lrm: {vname} = {v:.3}"));
                }
                ui.separator();
                for field_name in [
                    "gravity",
                    "spring",
                    "friction",
                    "friction_x",
                    "friction_y",
                    "scale_x",
                    "scale_y",
                ] {
                    if let Some(v) = def.fields.get(field_name).and_then(|v| v.as_f32()) {
                        ui.label(format!("{field_name} = {v}"));
                    }
                }
                for field_name in ["b_rate", "bend_spd", "bend_vol", "v_bound", "ud_eft"] {
                    if let Some(v) = def.fields.get(field_name) {
                        ui.label(format!("{field_name} = {:?} [unverified]", v.as_f32()));
                    }
                }
            });
    }
}

fn panel_layers(ui: &mut egui::Ui, model: &mut LoadedModel, state: &mut UiState) {
    ui.horizontal(|ui| {
        ui.label("Search:");
        ui.text_edit_singleline(&mut state.layer_filter);
        if ui.small_button("x").clicked() {
            state.layer_filter.clear();
        }
        ui.checkbox(&mut state.layer_filter_face_only, "Face only");
        ui.checkbox(&mut state.layer_filter_visible_only, "Visible only");
    });

    let filter = state.layer_filter.to_lowercase();

    egui::ScrollArea::vertical()
        .max_height(300.0)
        .show(ui, |ui| {
            for sprite in &model.scene.sprites {
                let fi = &sprite.draw_frame_info;
                let face = is_face_related_sprite(sprite);

                if state.layer_filter_face_only && !face {
                    continue;
                }
                if state.layer_filter_visible_only && (!sprite.visible || sprite.opacity <= 0.0) {
                    continue;
                }
                if !filter.is_empty() {
                    let path_lower = fi.path.to_lowercase();
                    let label_lower = fi.layer_label.as_deref().unwrap_or("").to_lowercase();
                    if !path_lower.contains(&filter) && !label_lower.contains(&filter) {
                        continue;
                    }
                }

                let color = if !sprite.visible || sprite.opacity <= 0.0 {
                    egui::Color32::DARK_GRAY
                } else if face {
                    egui::Color32::LIGHT_BLUE
                } else {
                    egui::Color32::WHITE
                };

                let label_short = fi.layer_label.as_deref().unwrap_or("?");
                let path = fi.path.as_str();
                let line = format!(
                    "di={:>3} z={:>6.2} {:<24} op={:.2} v={} tex={} pass={:?}",
                    fi.draw_index,
                    sprite.z,
                    label_short.chars().take(24).collect::<String>(),
                    sprite.opacity,
                    sprite.visible as u8,
                    sprite.texture_resource_index,
                    fi.pass,
                );

                let response = ui.selectable_label(
                    state.selected_layer.as_deref() == Some(path),
                    egui::RichText::new(line).color(color).monospace(),
                );
                if response.clicked() {
                    state.selected_layer = Some(fi.path.clone());
                }
                response.on_hover_text(path);
            }
        });

    // Selected layer details
    if let Some(sel) = &state.selected_layer.clone() {
        if let Some(sprite) = model
            .scene
            .sprites
            .iter()
            .find(|s| s.draw_frame_info.path == *sel)
        {
            ui.separator();
            ui.heading("Selected:");
            ui.label(format!("path: {}", sprite.draw_frame_info.path));
            ui.label(format!("label: {:?}", sprite.draw_frame_info.layer_label));
            ui.label(format!(
                "tex={} size={}x{}",
                sprite.texture_resource_index, sprite.width as u32, sprite.height as u32
            ));
            ui.label(format!(
                "center=({:.1},{:.1}) z={:.2}",
                sprite.center_x, sprite.center_y, sprite.z
            ));
            ui.label(format!(
                "uv: ({:.3},{:.3})->({:.3},{:.3})",
                sprite.uv_left, sprite.uv_top, sprite.uv_right, sprite.uv_bottom
            ));
            ui.label(format!(
                "stencil_type={} masks={}",
                sprite.draw_frame_info.stencil_type,
                sprite
                    .draw_frame_info
                    .stencil_composite_mask_layer_list
                    .len()
            ));
        }
    }
}

fn panel_textures(ui: &mut egui::Ui, model: &mut LoadedModel, _state: &mut UiState) {
    ui.heading("Texture Sources");
    for (name, tex) in &model.schema.textures {
        ui.label(format!(
            "[{}] {} {}x{} fmt={:?} compress={:?}",
            tex.resource_index,
            name,
            tex.width,
            tex.height,
            tex.format.as_deref(),
            tex.compress.as_deref()
        ));
        let face_icons: Vec<_> = tex
            .icons
            .iter()
            .filter(|(iname, _)| {
                let lower = iname.to_lowercase();
                lower.contains("eye")
                    || lower.contains("mouth")
                    || lower.contains("face")
                    || lower.contains("目")
                    || lower.contains("口")
                    || lower.contains("眉")
                    || lower.contains("鼻")
                    || lower.contains("輪郭")
            })
            .collect();
        if !face_icons.is_empty() {
            ui.indent(format!("icons_{name}"), |ui| {
                ui.label(format!("{} face icons:", face_icons.len()));
                for (iname, icon) in &face_icons {
                    ui.label(format!(
                        "  {} @ ({:.0},{:.0}) {}x{} ori=({:.1},{:.1})",
                        iname,
                        icon.left,
                        icon.top,
                        icon.resolved_width() as u32,
                        icon.resolved_height() as u32,
                        icon.origin_x,
                        icon.origin_y
                    ));
                }
            });
        }
    }

    ui.separator();
    ui.heading("Face UV refs");
    let face_sprites: Vec<_> = model
        .scene
        .sprites
        .iter()
        .filter(|s| is_face_related_sprite(s) && s.visible && s.opacity > 0.0)
        .collect();
    ui.label(format!("{} face sprites:", face_sprites.len()));
    egui::ScrollArea::vertical()
        .max_height(300.0)
        .show(ui, |ui| {
            for sprite in &face_sprites {
                let label = sprite
                    .draw_frame_info
                    .layer_label
                    .as_deref()
                    .unwrap_or_else(|| {
                        sprite
                            .draw_frame_info
                            .path
                            .rsplit('/')
                            .next()
                            .unwrap_or("?")
                    });
                ui.label(format!(
                    "{} tex={} uv:({:.3},{:.3})-({:.3},{:.3}) {}x{}",
                    label,
                    sprite.texture_resource_index,
                    sprite.uv_left,
                    sprite.uv_top,
                    sprite.uv_right,
                    sprite.uv_bottom,
                    sprite.width as u32,
                    sprite.height as u32,
                ));
            }
        });
}

fn preview_vertices() -> Result<Vec<GpuSpriteVertex>, Box<dyn Error>> {
    let positions = [
        [-0.6, -0.6],
        [0.0, -0.75],
        [0.6, -0.6],
        [-0.7, 0.0],
        [0.0, 0.0],
        [0.7, 0.0],
        [-0.6, 0.6],
        [0.0, 0.75],
        [0.6, 0.6],
    ];
    let colors = [
        0xffff_4040,
        0xffff_c040,
        0xffffff40,
        0xff40_ff40,
        0xffffffff,
        0xff40_ffff,
        0xff40_40ff,
        0xffc0_40ff,
        0xffff_40c0,
    ];

    let strips = build_d3d_triangle_strips(&positions, &colors, 0.0, 0.0, 2.0, 2.0, 2, 2, 2, 2)?;
    let triangle_list = expand_triangle_strips_to_list(&strips);
    Ok(triangle_list
        .into_iter()
        .map(GpuSpriteVertex::from_emote_preview)
        .collect())
}
