//! Runtime-facing Emote player state.
//!
//! This module keeps the public player/control surface alive while the original
//! mesh deformation path is still being recovered from `StepFrameMeshChain`.
//! Variable writes are accepted, queued, progressed, and queryable. Timeline
//! variable tracks from `metadata/timelineControl` are evaluated every frame and
//! feed the same variable map used by the renderer-side mesh deformation path.

use crate::api::{transform_order_mask, EmotePlayerControl, TimelinePlayMode, VariableWrite};
use crate::{
    load_emote_static_scene, EmoteSceneBounds, EmoteSchemaError, EmoteStaticScene, PsbFile,
    PsbValue,
};
use std::collections::BTreeMap;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(debug_assertions)]
static MIRROR_WARNING_PRINTED: AtomicBool = AtomicBool::new(false);

const PHYSICS_MAX_SUBSTEP_TICKS: f32 = 1.0;
const PHYSICS_EPSILON_TICKS: f32 = 0.00000011920929;
const PEND_BEND_POWER_STEP: f32 = 0.03125;
const PEND_BEND_TRIGGER_VALUE: f32 = 28.0;
const TAU: f32 = std::f32::consts::PI * 2.0;

const CONTROL_METADATA_KEYS: &[&str] = &[
    "bustControl",
    "hairControl",
    "partsControl",
    "eyeControl",
    "eyebrowControl",
    "mouthControl",
    "transitionControl",
    "selectorControl",
    "clampControl",
    "loopControl",
    "mirrorControl",
    "stereovisionControl",
];

fn is_auto_control_timeline_name(name: &str) -> bool {
    name.starts_with("@control/")
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmoteVariableFrameInfo {
    pub label: String,
    pub value: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmoteVariableInfo {
    pub name: String,
    pub default_value: f32,
    pub min_value: Option<f32>,
    pub max_value: Option<f32>,
    pub frames: Vec<EmoteVariableFrameInfo>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmoteVariableState {
    pub info: EmoteVariableInfo,
    pub value: f32,
    pub target: Option<EmoteVariableTarget>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmoteVariableTarget {
    pub start_value: f32,
    pub target_value: f32,
    pub elapsed_ticks: f32,
    pub duration_ticks: f32,
    pub easing: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmoteTimelineFrame {
    pub time_ticks: f32,
    pub value: f32,
    pub easing: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmoteTimelineVariable {
    pub name: String,
    pub frames: Vec<EmoteTimelineFrame>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmoteTimeline {
    pub name: String,
    pub path: Option<String>,
    pub duration_ticks: f32,
    pub variables: Vec<EmoteTimelineVariable>,
    pub is_difference: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct ActiveTimelineState {
    mode: TimelinePlayMode,
    elapsed_ticks: f32,
    blend_ratio: f32,
    blend_target: Option<TimelineBlendTarget>,
}

#[derive(Debug, Clone, PartialEq)]
struct TimelineBlendTarget {
    start_value: f32,
    target_value: f32,
    elapsed_ticks: f32,
    duration_ticks: f32,
    easing: f32,
    stop_when_done: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmoteApiLogEntry {
    pub command: String,
    pub args: Vec<String>,
}

impl EmoteApiLogEntry {
    fn new(command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            command: command.into(),
            args,
        }
    }

    fn encode(&self) -> String {
        if self.args.is_empty() {
            self.command.clone()
        } else {
            format!("{}\t{}", self.command, self.args.join("\t"))
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmoteCharaProfileInfo {
    pub label: String,
    pub value: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindState {
    pub start: f32,
    pub goal: f32,
    pub speed: f32,
    pub pow_min: f32,
    pub pow_max: f32,
    pub elapsed_ticks: f32,
}

/// Runtime state for one EPBustControl spring (one entry per bustControl item).
///
/// The original control keeps a root point, a bob point, a velocity vector,
/// and a root-to-target offset.  The group updater interpolates the target
/// baseLayer position over fixed substeps before calling EPBustControl::step.
#[derive(Debug, Clone, PartialEq)]
pub struct BustPhysicsState {
    /// Current bob position, corresponding to the original object at +52.
    pub bob: [f32; 3],
    /// Current bob velocity, corresponding to the original object at +64.
    pub vel: [f32; 3],
    /// Y rest/bias term used by the var_ud output.
    pub ofs: f32,
    /// Set on the first group update.
    pub first_tick: bool,
    /// Original root offset: root = target_baseLayer + root_offset.
    pub root_offset: [f32; 3],
    /// Previous baseLayer target, used for the original interpolated substep loop.
    pub last_anchor: Option<[f32; 3]>,
}

/// Runtime state for one EPPendControl two-segment pendulum.
///
/// The original control owns a root, two rest points, two current bob points,
/// two velocities, and a bend oscillator.  Hair and parts controls both use
/// this state type.
#[derive(Debug, Clone, PartialEq)]
pub struct HairPhysicsState {
    /// Current bob position for the two segments, original +112 and +124.
    pub bob: [[f32; 3]; 2],
    /// Current velocity for the two segments, original +136 and +148.
    pub vel: [[f32; 3]; 2],
    /// Y rest/bias term used by var_ud.
    pub ofs: f32,
    /// Set on the first group update.
    pub first_tick: bool,
    /// Original root offset: root = target_baseLayer + root_offset.
    pub root_offset: [f32; 3],
    /// Previous baseLayer target, used for the original interpolated substep loop.
    pub last_anchor: Option<[f32; 3]>,
    /// Bend oscillator phase, original field +164.
    pub bend_phase: f32,
    /// Bend oscillator power, original field +168.
    pub bend_power: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElunaPlayer {
    shown: bool,
    smoothing: bool,
    mesh_division_ratio: f32,
    queuing: bool,
    color_rgba: u32,
    grayscale: f32,
    as_original_scale: bool,
    coord: [f32; 2],
    scale: f32,
    rot: f32,
    elapsed_ticks: f32,
    pub paused: bool,
    pub physics_enabled: bool,
    scene: EmoteStaticScene,
    variables: BTreeMap<String, EmoteVariableState>,
    timelines: BTreeMap<String, EmoteTimeline>,
    pending_writes: Vec<VariableWrite>,
    active_timelines: BTreeMap<String, TimelinePlayMode>,
    active_timeline_states: BTreeMap<String, ActiveTimelineState>,
    timeline_diff_variables: BTreeMap<String, BTreeMap<String, EmoteVariableState>>,
    timeline_blend_ratios: BTreeMap<String, f32>,
    runtime_pipeline: EmoteRuntimePipeline,
    bust_states: Vec<BustPhysicsState>,
    hair_states: Vec<HairPhysicsState>,
    outer_forces: BTreeMap<String, [f32; 2]>,
    outer_rot: f32,
    transform_order_mask: u32,
    hair_scale: f32,
    parts_scale: f32,
    bust_scale: f32,
    wind: Option<WindState>,
    modified: bool,
    recording_api_log: bool,
    replaying_api_log: bool,
    api_log: Vec<EmoteApiLogEntry>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct EmoteRuntimePipeline {
    pub instant_variables: Vec<String>,
    pub selector_controls: Vec<SelectorControl>,
    pub clamp_controls: Vec<ClampControl>,
    pub loop_controls: Vec<LoopControl>,
    pub mirror_control: Option<MirrorControl>,
    pub transition_controls: Vec<TransitionControl>,
    pub physics_controls: Vec<PhysicsControl>,
    pub parts_controls: Vec<OpaqueControl>,
    pub eye_controls: Vec<OpaqueControl>,
    pub eyebrow_controls: Vec<OpaqueControl>,
    pub mouth_controls: Vec<OpaqueControl>,
    pub unsupported_fields: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectorControl {
    pub label: String,
    pub enabled: bool,
    pub option_list: Vec<SelectorOption>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectorOption {
    pub label: String,
    pub off_value: f32,
    pub on_value: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClampControl {
    pub label: String,
    pub enabled: bool,
    pub kind: i64,
    pub var_lr: String,
    pub var_ud: String,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoopControl {
    pub label: Option<String>,
    pub enabled: bool,
    pub var_loop: Option<String>,
    pub transition_list: Vec<LoopTransition>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoopTransition {
    pub start: f32,
    pub end: f32,
    pub duration_ticks: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MirrorControl {
    pub variable_match_list: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransitionControl {
    pub label: String,
    pub enabled: bool,
    pub fade: Option<f32>,
    pub diff: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhysicsControl {
    Bust(PhysicsControlDefinition),
    Hair(PhysicsControlDefinition),
    Parts(PhysicsControlDefinition),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsControlDefinition {
    pub label: String,
    pub enabled: bool,
    pub base_layer: Option<String>,
    pub parameter: Option<String>,
    pub var_lr: Option<String>,
    pub var_ud: Option<String>,
    pub var_lrm: Option<String>,
    pub fields: BTreeMap<String, PsbValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpaqueControl {
    pub label: Option<String>,
    pub enabled: bool,
    pub fields: BTreeMap<String, PsbValue>,
}

impl ElunaPlayer {
    pub fn from_psb(psb: &PsbFile) -> Result<Self, EmoteSchemaError> {
        let (_schema, scene) = load_emote_static_scene(psb)?;
        Ok(Self::from_scene_variables_timelines_runtime(
            scene,
            collect_emote_variables(psb),
            collect_emote_timelines(psb),
            collect_emote_runtime_pipeline(psb),
        ))
    }

    pub fn from_scene(scene: EmoteStaticScene) -> Self {
        Self::from_scene_and_variables(scene, Vec::new())
    }

    pub fn scene(&self) -> &EmoteStaticScene {
        &self.scene
    }

    pub fn replace_scene(&mut self, scene: EmoteStaticScene) {
        self.scene = scene;
    }

    pub fn bounds(&self) -> Option<EmoteSceneBounds> {
        self.scene.bounds
    }

    pub fn is_shown(&self) -> bool {
        self.shown
    }

    pub fn smoothing(&self) -> bool {
        self.smoothing
    }
    pub fn set_smoothing(&mut self, state: bool) {
        self.smoothing = state;
        self.modified = true;
    }

    pub fn mesh_division_ratio(&self) -> f32 {
        self.mesh_division_ratio
    }
    pub fn set_mesh_division_ratio(&mut self, ratio: f32) {
        if ratio.is_finite() && ratio > 0.0 {
            self.mesh_division_ratio = ratio;
            self.modified = true;
        }
    }

    pub fn queuing(&self) -> bool {
        self.queuing
    }
    pub fn set_queuing(&mut self, state: bool) {
        self.queuing = state;
        self.modified = true;
    }

    pub fn color_rgba(&self) -> u32 {
        self.color_rgba
    }
    pub fn set_color_rgba(&mut self, rgba: u32, _frame_count: f32, _easing: f32) {
        self.color_rgba = rgba;
        self.modified = true;
        self.record_api_call(
            "SetColor",
            vec![
                rgba.to_string(),
                _frame_count.max(0.0).to_string(),
                _easing.to_string(),
            ],
        );
    }

    pub fn grayscale(&self) -> f32 {
        self.grayscale
    }
    pub fn set_grayscale(&mut self, rate: f32, _frame_count: f32, _easing: f32) {
        if rate.is_finite() {
            self.grayscale = rate.clamp(0.0, 1.0);
            self.modified = true;
            self.record_api_call(
                "SetGrayscale",
                vec![
                    self.grayscale.to_string(),
                    _frame_count.max(0.0).to_string(),
                    _easing.to_string(),
                ],
            );
        }
    }

    pub fn as_original_scale(&self) -> bool {
        self.as_original_scale
    }
    pub fn set_as_original_scale(&mut self, state: bool) {
        self.as_original_scale = state;
        self.modified = true;
    }

    pub fn state_value(&self, label: &str) -> f32 {
        match label {
            "scale" => self.scale,
            "rot" => self.rot,
            "x" => self.coord[0],
            "y" => self.coord[1],
            _ => self.variable_value(label).unwrap_or(0.0),
        }
    }

    pub fn elapsed_ticks(&self) -> f32 {
        self.elapsed_ticks
    }

    pub fn variables(&self) -> &BTreeMap<String, EmoteVariableState> {
        &self.variables
    }

    pub fn timelines(&self) -> &BTreeMap<String, EmoteTimeline> {
        &self.timelines
    }

    pub fn default_timeline_name(&self) -> Option<&str> {
        self.timelines.keys().next().map(String::as_str)
    }

    pub fn main_timeline_labels(&self) -> Vec<&str> {
        self.timelines
            .values()
            .filter(|timeline| !timeline.name.starts_with("@control/") && !timeline.is_difference)
            .map(|timeline| timeline.name.as_str())
            .collect()
    }

    pub fn diff_timeline_labels(&self) -> Vec<&str> {
        self.timelines
            .values()
            .filter(|timeline| !timeline.name.starts_with("@control/") && timeline.is_difference)
            .map(|timeline| timeline.name.as_str())
            .collect()
    }

    pub fn playing_timeline_info(&self) -> Vec<(String, u32)> {
        self.active_timelines
            .iter()
            .map(|(name, mode)| (name.clone(), mode.flags))
            .collect()
    }

    pub fn timeline_total_frame_count(&self, name: &str) -> Option<f32> {
        self.timelines
            .get(name)
            .map(|timeline| timeline.duration_ticks)
    }

    pub fn timeline_blend_ratio(&self, name: &str) -> f32 {
        self.active_timeline_states
            .get(name)
            .map(|state| state.blend_ratio)
            .or_else(|| self.timeline_blend_ratios.get(name).copied())
            .unwrap_or(0.0)
    }

    pub fn is_timeline_playing(&self, name: &str) -> bool {
        if name.is_empty() {
            return !self.active_timelines.is_empty();
        }
        self.active_timelines.contains_key(name)
    }

    pub fn is_loop_timeline(&self, name: &str) -> bool {
        self.active_timeline_states
            .get(name)
            .map(|state| state.mode.is_looping())
            .or_else(|| {
                self.active_timelines
                    .get(name)
                    .map(|mode| mode.is_looping())
            })
            .unwrap_or(false)
    }

    pub fn variable_value(&self, name: &str) -> Option<f32> {
        self.variables.get(name).map(|state| state.value)
    }

    pub fn variable_frame_count(&self, name: &str) -> usize {
        self.variables
            .get(name)
            .map(|state| state.info.frames.len())
            .unwrap_or(0)
    }

    pub fn variable_frame_label_at(&self, name: &str, index: usize) -> Option<&str> {
        self.variables
            .get(name)?
            .info
            .frames
            .get(index)
            .map(|frame| frame.label.as_str())
    }

    pub fn variable_frame_value_at(&self, name: &str, index: usize) -> Option<f32> {
        self.variables
            .get(name)?
            .info
            .frames
            .get(index)
            .map(|frame| frame.value)
    }

    pub fn pending_writes(&self) -> &[VariableWrite] {
        &self.pending_writes
    }

    pub fn active_timelines(&self) -> &BTreeMap<String, TimelinePlayMode> {
        &self.active_timelines
    }

    pub fn apply_write(&mut self, write: VariableWrite) {
        self.set_variable_timed(&write.name, write.value, write.time_ticks, write.easing);
    }

    pub fn from_scene_and_variables(
        scene: EmoteStaticScene,
        infos: Vec<EmoteVariableInfo>,
    ) -> Self {
        Self::from_scene_variables_timelines(scene, infos, Vec::new())
    }

    pub fn from_scene_variables_timelines(
        scene: EmoteStaticScene,
        infos: Vec<EmoteVariableInfo>,
        timelines: Vec<EmoteTimeline>,
    ) -> Self {
        Self::from_scene_variables_timelines_runtime(
            scene,
            infos,
            timelines,
            EmoteRuntimePipeline::default(),
        )
    }

    pub fn from_scene_variables_timelines_runtime(
        scene: EmoteStaticScene,
        infos: Vec<EmoteVariableInfo>,
        timelines: Vec<EmoteTimeline>,
        runtime_pipeline: EmoteRuntimePipeline,
    ) -> Self {
        let mut variables = BTreeMap::new();
        for info in infos {
            let name = info.name.clone();
            variables.entry(name).or_insert_with(|| EmoteVariableState {
                value: info.default_value,
                info,
                target: None,
            });
        }

        let mut timeline_map = BTreeMap::new();
        for timeline in timelines {
            for variable in &timeline.variables {
                let first_value = variable.frames.first().map(|f| f.value).unwrap_or(0.0);
                variables
                    .entry(variable.name.clone())
                    .or_insert_with(|| EmoteVariableState {
                        info: EmoteVariableInfo {
                            name: variable.name.clone(),
                            default_value: first_value,
                            min_value: None,
                            max_value: None,
                            frames: Vec::new(),
                        },
                        value: first_value,
                        target: None,
                    });
                if let Some(state) = variables.get_mut(&variable.name) {
                    merge_timeline_variable_range(&mut state.info, variable);
                    if (state.value - state.info.default_value).abs() <= f32::EPSILON {
                        state.value = clamp_variable_value(&state.info, first_value);
                    }
                }
            }
            timeline_map.insert(timeline.name.clone(), timeline);
        }

        let mut active_timelines = BTreeMap::new();
        let mut active_timeline_states = BTreeMap::new();
        for (name, timeline) in &timeline_map {
            if is_auto_control_timeline_name(name) {
                active_timelines.insert(name.clone(), TimelinePlayMode::Loop);
                active_timeline_states.insert(
                    name.clone(),
                    ActiveTimelineState {
                        mode: TimelinePlayMode::Loop,
                        elapsed_ticks: 0.0,
                        blend_ratio: 1.0,
                        blend_target: None,
                    },
                );
                for variable in &timeline.variables {
                    let value = evaluate_timeline_variable(variable, 0.0);
                    let state = variables.entry(variable.name.clone()).or_insert_with(|| {
                        EmoteVariableState {
                            info: EmoteVariableInfo {
                                name: variable.name.clone(),
                                default_value: value,
                                min_value: None,
                                max_value: None,
                                frames: Vec::new(),
                            },
                            value,
                            target: None,
                        }
                    });
                    merge_timeline_variable_range(&mut state.info, variable);
                    state.value = clamp_variable_value(&state.info, value);
                    state.target = None;
                }
            }
        }

        let bust_states = init_bust_states(&runtime_pipeline);
        let hair_states = init_hair_states(&runtime_pipeline);

        let mut player = Self {
            shown: true,
            smoothing: true,
            mesh_division_ratio: 1.0,
            queuing: false,
            color_rgba: 0xffff_ffff,
            grayscale: 0.0,
            as_original_scale: false,
            coord: [0.0, 0.0],
            scale: 1.0,
            rot: 0.0,
            elapsed_ticks: 0.0,
            paused: false,
            physics_enabled: true,
            scene,
            variables,
            timelines: timeline_map,
            pending_writes: Vec::new(),
            active_timelines,
            active_timeline_states,
            timeline_diff_variables: BTreeMap::new(),
            timeline_blend_ratios: BTreeMap::new(),
            runtime_pipeline,
            bust_states,
            hair_states,
            outer_forces: BTreeMap::new(),
            outer_rot: 0.0,
            transform_order_mask: transform_order_mask::DEFAULT,
            hair_scale: 1.0,
            parts_scale: 1.0,
            bust_scale: 1.0,
            wind: None,
            modified: false,
            recording_api_log: false,
            replaying_api_log: false,
            api_log: Vec::new(),
        };
        player.evaluate_runtime_pipeline(0.0);
        player
    }

    pub fn runtime_pipeline(&self) -> &EmoteRuntimePipeline {
        &self.runtime_pipeline
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn set_physics_enabled(&mut self, enabled: bool) {
        self.physics_enabled = enabled;
    }

    pub fn is_physics_enabled(&self) -> bool {
        self.physics_enabled
    }

    pub fn evaluate_physics_for_current_scene(&mut self, delta_ticks: f32) {
        if !self.physics_enabled || !delta_ticks.is_finite() || delta_ticks < 0.0 {
            return;
        }
        self.evaluate_physics_controls(delta_ticks);
        self.modified = true;
    }

    pub fn progress_ticks_without_physics(&mut self, delta_ticks: f32) {
        self.progress_ticks_internal(delta_ticks, false);
    }

    pub fn set_variable_immediate(&mut self, name: &str, value: f32) {
        if let Some(state) = self.variables.get_mut(name) {
            state.value = clamp_variable_value(&state.info, value);
            state.target = None;
            self.modified = true;
            self.evaluate_runtime_pipeline(0.0);
        }
    }

    pub fn reset_variable_to_default(&mut self, name: &str) {
        if let Some(state) = self.variables.get_mut(name) {
            let default = state.info.default_value;
            state.value = default;
            state.target = None;
            self.modified = true;
        }
    }

    pub fn reset_physics(&mut self) {
        self.bust_states = init_bust_states(&self.runtime_pipeline);
        self.hair_states = init_hair_states(&self.runtime_pipeline);
    }

    pub fn timeline_elapsed_ticks(&self, name: &str) -> f32 {
        self.active_timeline_states
            .get(name)
            .map(|s| s.elapsed_ticks)
            .unwrap_or(0.0)
    }

    pub fn set_timeline_time(&mut self, name: &str, ticks: f32) -> Result<(), String> {
        let Some(timeline) = self.timelines.get(name).cloned() else {
            return Err(format!("timeline not found: {name}"));
        };
        let duration = timeline.duration_ticks.max(0.0);
        let local_time = if duration > 0.0 {
            ticks.clamp(0.0, duration)
        } else {
            ticks.max(0.0)
        };
        let mode = self
            .active_timeline_states
            .get(name)
            .map(|state| state.mode)
            .or_else(|| self.active_timelines.get(name).copied())
            .unwrap_or(TimelinePlayMode::Once);
        self.active_timelines.insert(name.to_owned(), mode);
        let blend_ratio = self.timeline_blend_ratios.get(name).copied().unwrap_or(1.0);
        self.active_timeline_states.insert(
            name.to_owned(),
            ActiveTimelineState {
                mode,
                elapsed_ticks: local_time,
                blend_ratio,
                blend_target: None,
            },
        );
        self.reapply_active_timelines_at_current_time();
        self.evaluate_runtime_pipeline(0.0);
        Ok(())
    }

    pub fn set_selector_option(&mut self, label: &str, option_index: usize) -> Result<(), String> {
        let Some(control) = self
            .runtime_pipeline
            .selector_controls
            .iter()
            .find(|control| control.label == label)
        else {
            return Err(format!("selectorControl not found: {label}"));
        };
        if option_index >= control.option_list.len() {
            return Err(format!(
                "selectorControl {label} option index {option_index} out of range {}",
                control.option_list.len()
            ));
        }
        self.set_variable_immediate(label, option_index as f32);
        self.evaluate_runtime_pipeline(0.0);
        Ok(())
    }

    pub fn reset_all_variables_to_default(&mut self) {
        for state in self.variables.values_mut() {
            state.value = state.info.default_value;
            state.target = None;
        }
        self.modified = true;
        self.evaluate_runtime_pipeline(0.0);
    }

    fn set_variable_timed_internal(
        &mut self,
        name: &str,
        value: f32,
        time_ticks: f32,
        easing: f32,
    ) -> Option<f32> {
        if name.is_empty() || !value.is_finite() {
            return None;
        }

        let state = self.ensure_variable(name);
        let mut target_value = value;
        if let Some(min) = state.info.min_value {
            target_value = target_value.max(min);
        }
        if let Some(max) = state.info.max_value {
            target_value = target_value.min(max);
        }

        if time_ticks <= 0.0 || !time_ticks.is_finite() {
            state.value = target_value;
            state.target = None;
        } else {
            state.target = Some(EmoteVariableTarget {
                start_value: state.value,
                target_value,
                elapsed_ticks: 0.0,
                duration_ticks: time_ticks,
                easing,
            });
        }

        self.pending_writes.push(VariableWrite::timed(
            name,
            target_value,
            time_ticks.max(0.0),
            easing,
        ));
        self.modified = true;
        Some(target_value)
    }

    pub fn variable_diff_value(&self, module: &str, name: &str) -> Option<f32> {
        self.variable_value(&format!("{module}/{name}"))
    }

    pub fn is_animating(&self) -> bool {
        if self
            .active_timeline_states
            .values()
            .any(|state| state.blend_ratio > 0.0)
        {
            return true;
        }
        self.variables.values().any(|state| state.target.is_some()) || self.wind.is_some()
    }

    pub fn is_modified(&self) -> bool {
        self.modified
    }

    pub fn clear_modified(&mut self) {
        self.modified = false;
    }

    pub fn pass(&mut self) {
        self.reapply_active_timelines_at_current_time();
        self.evaluate_runtime_pipeline(0.0);
        self.modified = true;
        self.record_api_call("Pass", Vec::new());
    }

    pub fn step(&mut self) {
        <Self as EmotePlayerControl>::progress_ticks(self, 1.0);
        self.record_api_call("Step", Vec::new());
    }

    pub fn start_record_api_log(&mut self) {
        self.api_log.clear();
        self.recording_api_log = true;
    }

    pub fn stop_record_api_log(&mut self) {
        self.recording_api_log = false;
    }

    pub fn is_recording_api_log(&self) -> bool {
        self.recording_api_log
    }

    pub fn start_replay_api_log(&mut self) {
        self.replaying_api_log = true;
    }

    pub fn stop_replay_api_log(&mut self) {
        self.replaying_api_log = false;
    }

    pub fn is_replaying_api_log(&self) -> bool {
        self.replaying_api_log
    }

    pub fn clear_api_log(&mut self) {
        self.api_log.clear();
    }

    pub fn api_log(&self) -> String {
        self.api_log
            .iter()
            .map(EmoteApiLogEntry::encode)
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn set_api_log(&mut self, log: &str) {
        self.api_log = log.lines().filter_map(parse_api_log_entry).collect();
    }

    pub fn replay_api_log_once(&mut self) {
        let entries = self.api_log.clone();
        for entry in entries {
            self.apply_api_log_entry(&entry);
        }
    }

    fn record_api_call(&mut self, command: &str, args: Vec<String>) {
        if self.recording_api_log && !self.replaying_api_log {
            self.api_log.push(EmoteApiLogEntry::new(command, args));
        }
    }

    fn apply_api_log_entry(&mut self, entry: &EmoteApiLogEntry) {
        self.replaying_api_log = true;
        match entry.command.as_str() {
            "SetVariable" if entry.args.len() >= 4 => {
                if let (Ok(value), Ok(frame_count), Ok(easing)) = (
                    entry.args[1].parse::<f32>(),
                    entry.args[2].parse::<f32>(),
                    entry.args[3].parse::<f32>(),
                ) {
                    self.set_variable_timed(&entry.args[0], value, frame_count, easing);
                }
            }
            "SetVariableDiff" if entry.args.len() >= 5 => {
                if let (Ok(value), Ok(frame_count), Ok(easing)) = (
                    entry.args[2].parse::<f32>(),
                    entry.args[3].parse::<f32>(),
                    entry.args[4].parse::<f32>(),
                ) {
                    self.set_variable_diff(
                        &entry.args[0],
                        &entry.args[1],
                        value,
                        frame_count,
                        easing,
                    );
                }
            }
            "PlayTimeline" if entry.args.len() >= 2 => {
                if let Ok(flags) = entry.args[1].parse::<u32>() {
                    self.play_timeline(
                        &entry.args[0],
                        TimelinePlayMode {
                            flags,
                            looping: false,
                        },
                    );
                }
            }
            "StopTimeline" if !entry.args.is_empty() => self.stop_timeline(&entry.args[0]),
            "SetTimelineBlendRatio" if entry.args.len() >= 5 => {
                if let (Ok(value), Ok(frame_count), Ok(easing), Ok(stop_when_done)) = (
                    entry.args[1].parse::<f32>(),
                    entry.args[2].parse::<f32>(),
                    entry.args[3].parse::<f32>(),
                    entry.args[4].parse::<bool>(),
                ) {
                    self.set_timeline_blend_ratio(
                        &entry.args[0],
                        value,
                        frame_count,
                        easing,
                        stop_when_done,
                    );
                }
            }
            "SetOuterForce" if entry.args.len() >= 5 => {
                if let (Ok(x), Ok(y), Ok(frame_count), Ok(easing)) = (
                    entry.args[1].parse::<f32>(),
                    entry.args[2].parse::<f32>(),
                    entry.args[3].parse::<f32>(),
                    entry.args[4].parse::<f32>(),
                ) {
                    self.set_outer_force(&entry.args[0], x, y, frame_count, easing);
                }
            }
            "SetOuterRot" if entry.args.len() >= 3 => {
                if let (Ok(rot), Ok(frame_count), Ok(easing)) = (
                    entry.args[0].parse::<f32>(),
                    entry.args[1].parse::<f32>(),
                    entry.args[2].parse::<f32>(),
                ) {
                    self.set_outer_rot(rot, frame_count, easing);
                }
            }
            "SetCoord" if entry.args.len() >= 2 => {
                if let (Ok(x), Ok(y)) = (entry.args[0].parse::<f32>(), entry.args[1].parse::<f32>())
                {
                    self.set_coord(x, y);
                }
            }
            "SetScale" if !entry.args.is_empty() => {
                if let Ok(scale) = entry.args[0].parse::<f32>() {
                    self.set_scale(scale);
                }
            }
            "SetRot" if !entry.args.is_empty() => {
                if let Ok(rot) = entry.args[0].parse::<f32>() {
                    self.set_rot(rot);
                }
            }
            "Progress" if !entry.args.is_empty() => {
                if let Ok(frame_count) = entry.args[0].parse::<f32>() {
                    <Self as EmotePlayerControl>::progress_ticks(self, frame_count);
                }
            }
            _ => {}
        }
        self.replaying_api_log = false;
    }

    fn ensure_variable(&mut self, name: &str) -> &mut EmoteVariableState {
        self.variables
            .entry(name.to_owned())
            .or_insert_with(|| EmoteVariableState {
                info: EmoteVariableInfo {
                    name: name.to_owned(),
                    default_value: 0.0,
                    min_value: None,
                    max_value: None,
                    frames: Vec::new(),
                },
                value: 0.0,
                target: None,
            })
    }

    fn reset_timeline_variables_to_default(&mut self) {
        let mut names = Vec::new();
        for timeline in self.timelines.values() {
            for variable in &timeline.variables {
                names.push(variable.name.clone());
            }
        }
        names.sort();
        names.dedup();
        for name in names {
            if let Some(state) = self.variables.get_mut(&name) {
                state.value = state.info.default_value;
                state.target = None;
            }
        }
    }

    fn reapply_active_timelines_at_current_time(&mut self) {
        let mut applications = Vec::new();
        for (name, state) in self.active_timeline_states.clone() {
            let Some(timeline) = self.timelines.get(&name).cloned() else {
                continue;
            };
            applications.push((timeline, state.elapsed_ticks, state.mode, state.blend_ratio));
        }
        applications.sort_by(|a, b| {
            a.2.is_difference()
                .cmp(&b.2.is_difference())
                .then_with(|| a.0.name.cmp(&b.0.name))
        });
        self.reset_timeline_variables_to_default();
        for (timeline, local_time, mode, blend_ratio) in applications {
            self.apply_timeline_at(&timeline, local_time, mode, blend_ratio);
        }
    }

    fn progress_active_timelines(&mut self, delta_ticks: f32) {
        let names: Vec<String> = self.active_timeline_states.keys().cloned().collect();
        let mut applications = Vec::new();
        for name in names {
            let Some(timeline) = self.timelines.get(&name).cloned() else {
                self.active_timeline_states.remove(&name);
                self.active_timelines.remove(&name);
                continue;
            };
            let Some(state) = self.active_timeline_states.get_mut(&name) else {
                continue;
            };

            state.elapsed_ticks += delta_ticks;
            let mut local_time = state.elapsed_ticks;
            let duration = timeline.duration_ticks.max(0.0);
            if duration > 0.0 && local_time > duration {
                if !state.mode.is_looping() {
                    local_time = duration;
                    state.elapsed_ticks = duration;
                } else {
                    local_time = local_time.rem_euclid(duration);
                    state.elapsed_ticks = local_time;
                }
            }

            let mode = state.mode;
            let blend_ratio = advance_timeline_blend(state, delta_ticks);
            self.timeline_blend_ratios.insert(name.clone(), blend_ratio);
            applications.push((timeline, local_time, mode, blend_ratio));
        }

        applications.sort_by(|a, b| {
            a.2.is_difference()
                .cmp(&b.2.is_difference())
                .then_with(|| a.0.name.cmp(&b.0.name))
        });
        self.reset_timeline_variables_to_default();
        for (timeline, local_time, mode, blend_ratio) in applications {
            self.apply_timeline_at(&timeline, local_time, mode, blend_ratio);
        }
    }

    fn apply_timeline_at(
        &mut self,
        timeline: &EmoteTimeline,
        local_time: f32,
        mode: TimelinePlayMode,
        blend_ratio: f32,
    ) {
        let blend_ratio = blend_ratio.clamp(0.0, 1.0);
        for variable in &timeline.variables {
            let evaluated = evaluate_timeline_variable(variable, local_time);
            let state = self.ensure_variable(&variable.name);
            let value = if mode.is_difference() {
                state.value + (evaluated - state.info.default_value) * blend_ratio
            } else {
                state.info.default_value + (evaluated - state.info.default_value) * blend_ratio
            };
            state.value = clamp_variable_value(&state.info, value);
            state.target = None;
        }
    }

    fn evaluate_runtime_pipeline(&mut self, delta_ticks: f32) {
        let pipeline = self.runtime_pipeline.clone();
        // selectorControl is the only top-level control whose PSB schema is
        // really an optionList of {label, offValue, onValue}.  eyeControl,
        // eyebrowControl, mouthControl, partsControl, hairControl, bustControl
        // have their own fields per the reverse notes (blink timers, talk
        // labels, EPPendControl / EPBustControl physics parameters).  Their
        // dedicated evaluators are not yet implemented; parts/hair/bust still
        // drive their variables through the physics block below, while
        // eye/eyebrow/mouth remain parsed but unevaluated rather than being
        // misapplied through a selector-style writer.
        for control in &pipeline.selector_controls {
            evaluate_selector_control(control, &mut self.variables);
        }
        for control in &pipeline.transition_controls {
            evaluate_transition_control(control, &mut self.variables);
        }
        for control in &pipeline.loop_controls {
            evaluate_loop_control(control, self.elapsed_ticks, &mut self.variables);
        }
        if let Some(control) = &pipeline.mirror_control {
            evaluate_mirror_control(control, &mut self.variables);
        }
        for control in &pipeline.clamp_controls {
            evaluate_clamp_control(control, &mut self.variables);
        }

        if self.physics_enabled {
            self.evaluate_physics_controls(delta_ticks);
        }
    }

    fn evaluate_physics_controls(&mut self, delta_ticks: f32) {
        if delta_ticks <= PHYSICS_EPSILON_TICKS {
            return;
        }
        let pipeline = self.runtime_pipeline.clone();
        let mut bust_idx = 0;
        let mut pend_idx = 0;
        for control in &pipeline.physics_controls {
            match control {
                PhysicsControl::Bust(def) => {
                    let (anchor, angle) = self.layer_world_pose(def.base_layer.as_deref());
                    let outer_force = self.outer_force("bust");
                    let scale = self.bust_scale;
                    if let Some(state) = self.bust_states.get_mut(bust_idx) {
                        step_bust_physics(
                            state,
                            def,
                            delta_ticks,
                            anchor,
                            angle + self.outer_rot,
                            outer_force,
                            scale,
                            &mut self.variables,
                        );
                    }
                    bust_idx += 1;
                }
                PhysicsControl::Hair(def) => {
                    let (anchor, angle) = self.layer_world_pose(def.base_layer.as_deref());
                    let outer_force = self.outer_force("hair");
                    let scale = self.hair_scale;
                    if let Some(state) = self.hair_states.get_mut(pend_idx) {
                        step_hair_physics(
                            state,
                            def,
                            delta_ticks,
                            anchor,
                            angle + self.outer_rot,
                            outer_force,
                            scale,
                            &mut self.variables,
                        );
                    }
                    pend_idx += 1;
                }
                PhysicsControl::Parts(def) => {
                    let (anchor, angle) = self.layer_world_pose(def.base_layer.as_deref());
                    let outer_force = self.outer_force("parts");
                    let scale = self.parts_scale;
                    if let Some(state) = self.hair_states.get_mut(pend_idx) {
                        step_hair_physics(
                            state,
                            def,
                            delta_ticks,
                            anchor,
                            angle + self.outer_rot,
                            outer_force,
                            scale,
                            &mut self.variables,
                        );
                    }
                    pend_idx += 1;
                }
            }
        }
    }

    fn progress_ticks_internal(&mut self, delta_ticks: f32, include_physics: bool) {
        if !delta_ticks.is_finite() || delta_ticks <= 0.0 {
            return;
        }
        self.elapsed_ticks += delta_ticks;

        if !self.paused {
            self.progress_active_timelines(delta_ticks);
        }
        if include_physics {
            self.evaluate_runtime_pipeline(delta_ticks);
        } else {
            let was_enabled = self.physics_enabled;
            self.physics_enabled = false;
            self.evaluate_runtime_pipeline(0.0);
            self.physics_enabled = was_enabled;
        }
        self.modified = true;

        for state in self.variables.values_mut() {
            let Some(mut target) = state.target.take() else {
                continue;
            };

            target.elapsed_ticks =
                (target.elapsed_ticks + delta_ticks).min(target.duration_ticks.max(0.0));
            let t = if target.duration_ticks <= 0.0 {
                1.0
            } else {
                (target.elapsed_ticks / target.duration_ticks).clamp(0.0, 1.0)
            };
            let eased = preview_easing(t, target.easing);
            let mut value = target.start_value + (target.target_value - target.start_value) * eased;
            if let Some(min) = state.info.min_value {
                value = value.max(min);
            }
            if let Some(max) = state.info.max_value {
                value = value.min(max);
            }
            state.value = value;

            if t < 1.0 {
                state.target = Some(target);
            }
        }
    }

    fn layer_world_pose(&self, base_layer: Option<&str>) -> ([f32; 3], f32) {
        let Some(base_layer) = base_layer.filter(|s| !s.is_empty()) else {
            return ([0.0, 0.0, 0.0], 0.0);
        };
        if let Some(layer) = self.scene.layer_states.iter().find(|layer| {
            layer.path == base_layer
                || layer.draw_frame_info.layer_label.as_deref() == Some(base_layer)
                || layer.path.ends_with(base_layer)
        }) {
            let m = layer.transform;
            let x = m[4];
            let y = m[5];
            let angle = m[2].atan2(m[0]);
            return ([x, y, 0.0], angle);
        }
        let sprite = self.scene.sprites.iter().find(|sprite| {
            sprite.draw_frame_info.path == base_layer
                || sprite.label.as_deref() == Some(base_layer)
                || sprite.draw_frame_info.layer_label.as_deref() == Some(base_layer)
                || sprite.draw_frame_info.path.ends_with(base_layer)
        });
        let Some(sprite) = sprite else {
            return ([0.0, 0.0, 0.0], 0.0);
        };
        let m = sprite.world_transform;
        let x = m[0] * sprite.center_x + m[1] * sprite.center_y + m[4];
        let y = m[2] * sprite.center_x + m[3] * sprite.center_y + m[5];
        let angle = m[2].atan2(m[0]);
        ([x, y, 0.0], angle)
    }
}

impl EmotePlayerControl for ElunaPlayer {
    fn show(&mut self) {
        self.shown = true;
    }

    fn hide(&mut self) {
        self.shown = false;
    }

    fn progress_ticks(&mut self, delta_ticks: f32) {
        self.progress_ticks_internal(delta_ticks, true);
    }

    fn render(&mut self) {}

    fn coord(&self) -> [f32; 2] {
        self.coord
    }

    fn set_coord(&mut self, x: f32, y: f32) {
        if x.is_finite() && y.is_finite() {
            self.coord = [x, y];
            self.modified = true;
            self.record_api_call(
                "SetCoord",
                vec![x.to_string(), y.to_string(), "0".to_owned(), "0".to_owned()],
            );
        }
    }

    fn scale(&self) -> f32 {
        self.scale
    }

    fn set_scale(&mut self, scale: f32) {
        if scale.is_finite() && scale > 0.0 {
            self.scale = scale;
            self.modified = true;
            self.record_api_call(
                "SetScale",
                vec![scale.to_string(), "0".to_owned(), "0".to_owned()],
            );
        }
    }

    fn rot(&self) -> f32 {
        self.rot
    }

    fn set_rot(&mut self, rot: f32) {
        if rot.is_finite() {
            self.rot = rot;
            self.modified = true;
            self.record_api_call(
                "SetRot",
                vec![rot.to_string(), "0".to_owned(), "0".to_owned()],
            );
        }
    }

    fn set_variable_timed(&mut self, name: &str, value: f32, time_ticks: f32, easing: f32) {
        if let Some(target_value) =
            self.set_variable_timed_internal(name, value, time_ticks, easing)
        {
            self.record_api_call(
                "SetVariable",
                vec![
                    name.to_owned(),
                    target_value.to_string(),
                    time_ticks.max(0.0).to_string(),
                    easing.to_string(),
                ],
            );
        }
    }

    fn set_variable_diff(
        &mut self,
        module: &str,
        name: &str,
        value: f32,
        time_ticks: f32,
        easing: f32,
    ) {
        if module.is_empty() || name.is_empty() {
            return;
        }
        let full_name = format!("{module}/{name}");
        if self
            .set_variable_timed_internal(&full_name, value, time_ticks, easing)
            .is_some()
        {
            self.record_api_call(
                "SetVariableDiff",
                vec![
                    module.to_owned(),
                    name.to_owned(),
                    value.to_string(),
                    time_ticks.max(0.0).to_string(),
                    easing.to_string(),
                ],
            );
        }
    }

    fn play_timeline(&mut self, name: &str, mode: TimelinePlayMode) {
        if !name.is_empty() {
            if !mode.is_difference() && !name.starts_with("@control/") {
                let old_main: Vec<String> = self
                    .active_timelines
                    .iter()
                    .filter(|(active_name, active_mode)| {
                        !active_name.starts_with("@control/")
                            && !active_mode.is_difference()
                            && active_name.as_str() != name
                    })
                    .map(|(active_name, _)| active_name.clone())
                    .collect();
                for old in old_main {
                    self.active_timelines.remove(&old);
                    self.active_timeline_states.remove(&old);
                    self.timeline_blend_ratios.remove(&old);
                }
            }

            self.active_timelines.insert(name.to_owned(), mode);
            let blend_ratio = self.timeline_blend_ratios.get(name).copied().unwrap_or(1.0);
            self.active_timeline_states.insert(
                name.to_owned(),
                ActiveTimelineState {
                    mode,
                    elapsed_ticks: 0.0,
                    blend_ratio,
                    blend_target: None,
                },
            );
            self.reapply_active_timelines_at_current_time();
            self.evaluate_runtime_pipeline(0.0);
            self.modified = true;
            self.record_api_call(
                "PlayTimeline",
                vec![name.to_owned(), mode.flags.to_string()],
            );
        }
    }

    fn stop_timeline(&mut self, name: &str) {
        if name.is_empty() {
            self.active_timelines.clear();
            self.active_timeline_states.clear();
            self.timeline_blend_ratios.clear();
        } else {
            self.active_timelines.remove(name);
            self.active_timeline_states.remove(name);
            self.timeline_blend_ratios.remove(name);
        }
        self.reapply_active_timelines_at_current_time();
        self.evaluate_runtime_pipeline(0.0);
        self.modified = true;
        self.record_api_call("StopTimeline", vec![name.to_owned()]);
    }

    fn set_timeline_blend_ratio(
        &mut self,
        name: &str,
        value: f32,
        time_ticks: f32,
        easing: f32,
        stop_when_done: bool,
    ) {
        if name.is_empty() || !value.is_finite() {
            return;
        }
        let target_value = value.clamp(0.0, 1.0);
        let next_ratio = {
            let entry = self
                .active_timeline_states
                .entry(name.to_owned())
                .or_insert_with(|| ActiveTimelineState {
                    mode: self
                        .active_timelines
                        .get(name)
                        .copied()
                        .unwrap_or(TimelinePlayMode::PARALLEL),
                    elapsed_ticks: 0.0,
                    blend_ratio: self.timeline_blend_ratios.get(name).copied().unwrap_or(1.0),
                    blend_target: None,
                });
            if time_ticks <= 0.0 || !time_ticks.is_finite() {
                entry.blend_ratio = target_value;
                entry.blend_target = None;
            } else {
                entry.blend_target = Some(TimelineBlendTarget {
                    start_value: entry.blend_ratio,
                    target_value,
                    elapsed_ticks: 0.0,
                    duration_ticks: time_ticks,
                    easing,
                    stop_when_done,
                });
            }
            entry.blend_ratio
        };
        self.timeline_blend_ratios
            .insert(name.to_owned(), next_ratio);
        self.modified = true;
        self.record_api_call(
            "SetTimelineBlendRatio",
            vec![
                name.to_owned(),
                target_value.to_string(),
                time_ticks.max(0.0).to_string(),
                easing.to_string(),
                stop_when_done.to_string(),
            ],
        );
    }

    fn fade_in_timeline(&mut self, name: &str, time_ticks: f32, easing: f32) {
        if !self.active_timelines.contains_key(name) {
            self.play_timeline(name, TimelinePlayMode::PARALLEL);
        }
        self.set_timeline_blend_ratio(name, 1.0, time_ticks, easing, false);
    }

    fn fade_out_timeline(&mut self, name: &str, time_ticks: f32, easing: f32) {
        self.set_timeline_blend_ratio(name, 0.0, time_ticks, easing, true);
    }

    fn set_outer_force(&mut self, label: &str, x: f32, y: f32, _time_ticks: f32, _easing: f32) {
        if label.is_empty() || !x.is_finite() || !y.is_finite() {
            return;
        }
        self.outer_forces.insert(label.to_owned(), [x, y]);
        self.modified = true;
        self.record_api_call(
            "SetOuterForce",
            vec![
                label.to_owned(),
                x.to_string(),
                y.to_string(),
                _time_ticks.max(0.0).to_string(),
                _easing.to_string(),
            ],
        );
    }

    fn outer_force(&self, label: &str) -> [f32; 2] {
        self.outer_forces.get(label).copied().unwrap_or([0.0, 0.0])
    }

    fn set_outer_rot(&mut self, rot: f32, _time_ticks: f32, _easing: f32) {
        if rot.is_finite() {
            self.outer_rot = rot;
            self.modified = true;
            self.record_api_call(
                "SetOuterRot",
                vec![
                    rot.to_string(),
                    _time_ticks.max(0.0).to_string(),
                    _easing.to_string(),
                ],
            );
        }
    }

    fn outer_rot(&self) -> f32 {
        self.outer_rot
    }

    fn start_wind(&mut self, start: f32, goal: f32, speed: f32, pow_min: f32, pow_max: f32) {
        self.wind = Some(WindState {
            start,
            goal,
            speed,
            pow_min,
            pow_max,
            elapsed_ticks: 0.0,
        });
        self.modified = true;
        self.record_api_call(
            "StartWind",
            vec![
                start.to_string(),
                goal.to_string(),
                speed.to_string(),
                pow_min.to_string(),
                pow_max.to_string(),
            ],
        );
    }

    fn stop_wind(&mut self) {
        self.wind = None;
        self.modified = true;
        self.record_api_call("StopWind", Vec::new());
    }

    fn set_transform_order_mask(&mut self, mask: u32) {
        self.transform_order_mask = mask;
    }

    fn transform_order_mask(&self) -> u32 {
        self.transform_order_mask
    }

    fn set_hair_scale(&mut self, scale: f32) {
        if scale.is_finite() && scale >= 0.0 {
            self.hair_scale = scale;
        }
    }

    fn hair_scale(&self) -> f32 {
        self.hair_scale
    }

    fn set_parts_scale(&mut self, scale: f32) {
        if scale.is_finite() && scale >= 0.0 {
            self.parts_scale = scale;
        }
    }

    fn parts_scale(&self) -> f32 {
        self.parts_scale
    }

    fn set_bust_scale(&mut self, scale: f32) {
        if scale.is_finite() && scale >= 0.0 {
            self.bust_scale = scale;
        }
    }

    fn bust_scale(&self) -> f32 {
        self.bust_scale
    }

    fn skip(&mut self) {
        for state in self.variables.values_mut() {
            if let Some(target) = state.target.take() {
                state.value = target.target_value;
            }
        }
    }
}

fn advance_timeline_blend(state: &mut ActiveTimelineState, delta_ticks: f32) -> f32 {
    let Some(mut target) = state.blend_target.take() else {
        return state.blend_ratio.clamp(0.0, 1.0);
    };

    target.elapsed_ticks =
        (target.elapsed_ticks + delta_ticks.max(0.0)).min(target.duration_ticks.max(0.0));
    let t = if target.duration_ticks <= 0.0 {
        1.0
    } else {
        (target.elapsed_ticks / target.duration_ticks).clamp(0.0, 1.0)
    };
    let eased = preview_easing(t, target.easing);
    state.blend_ratio =
        (target.start_value + (target.target_value - target.start_value) * eased).clamp(0.0, 1.0);
    if t < 1.0 {
        state.blend_target = Some(target);
    } else if target.stop_when_done && state.blend_ratio <= 0.0 {
        state.elapsed_ticks = 0.0;
    }
    state.blend_ratio
}

fn preview_easing(t: f32, easing: f32) -> f32 {
    if !easing.is_finite() || easing == 0.0 {
        return t;
    }

    // A conservative preview curve for interactive testing. The exact original
    // easing mapping still needs the shared SetVariable/easing path.
    if easing > 0.0 {
        t * t * (3.0 - 2.0 * t)
    } else {
        1.0 - (1.0 - t) * (1.0 - t)
    }
}

fn clamp_variable_value(info: &EmoteVariableInfo, mut value: f32) -> f32 {
    if let Some(min) = info.min_value {
        value = value.max(min);
    }
    if let Some(max) = info.max_value {
        value = value.min(max);
    }
    value
}

fn merge_timeline_variable_range(info: &mut EmoteVariableInfo, variable: &EmoteTimelineVariable) {
    for frame in &variable.frames {
        merge_range_value(&mut info.min_value, &mut info.max_value, frame.value);
    }
}

fn merge_range_value(min_value: &mut Option<f32>, max_value: &mut Option<f32>, value: f32) {
    if !value.is_finite() {
        return;
    }
    *min_value = Some(min_value.map_or(value, |min| min.min(value)));
    *max_value = Some(max_value.map_or(value, |max| max.max(value)));
}

fn evaluate_timeline_variable(variable: &EmoteTimelineVariable, time_ticks: f32) -> f32 {
    if variable.frames.is_empty() {
        return 0.0;
    }
    if time_ticks <= variable.frames[0].time_ticks {
        return variable.frames[0].value;
    }

    let mut prev = &variable.frames[0];
    for next in &variable.frames[1..] {
        if time_ticks <= next.time_ticks {
            let span = (next.time_ticks - prev.time_ticks).max(0.0);
            if span <= f32::EPSILON {
                return next.value;
            }
            let t = ((time_ticks - prev.time_ticks) / span).clamp(0.0, 1.0);
            let eased = preview_easing(t, next.easing);
            return prev.value + (next.value - prev.value) * eased;
        }
        prev = next;
    }
    prev.value
}

pub fn collect_emote_runtime_pipeline(psb: &PsbFile) -> EmoteRuntimePipeline {
    let mut pipeline = EmoteRuntimePipeline::default();
    let Some(metadata) = psb.root.field("metadata") else {
        return pipeline;
    };

    pipeline.instant_variables = metadata
        .field("instantVariableList")
        .and_then(PsbValue::as_list)
        .map(|items| {
            items
                .iter()
                .filter_map(PsbValue::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    pipeline.selector_controls = parse_selector_controls(metadata.field("selectorControl"));
    pipeline.clamp_controls = parse_clamp_controls(metadata.field("clampControl"));
    pipeline.loop_controls = parse_loop_controls(metadata.field("loopControl"));
    pipeline.mirror_control = parse_mirror_control(metadata.field("mirrorControl"));
    pipeline.transition_controls = parse_transition_controls(metadata.field("transitionControl"));
    pipeline.physics_controls.extend(parse_physics_controls(
        metadata.field("bustControl"),
        PhysicsControlKind::Bust,
    ));
    pipeline.physics_controls.extend(parse_physics_controls(
        metadata.field("hairControl"),
        PhysicsControlKind::Hair,
    ));
    pipeline.physics_controls.extend(parse_physics_controls(
        metadata.field("partsControl"),
        PhysicsControlKind::Parts,
    ));
    pipeline.parts_controls = parse_opaque_controls(metadata.field("partsControl"));
    pipeline.eye_controls = parse_opaque_controls(metadata.field("eyeControl"));
    pipeline.eyebrow_controls = parse_opaque_controls(metadata.field("eyebrowControl"));
    pipeline.mouth_controls = parse_opaque_controls(metadata.field("mouthControl"));

    if metadata.field("physicsVariableList").is_none() {
        pipeline
            .unsupported_fields
            .push("metadata.physicsVariableList is absent in this PSB".to_owned());
    }
    pipeline
}

fn parse_selector_controls(value: Option<&PsbValue>) -> Vec<SelectorControl> {
    value
        .and_then(PsbValue::as_list)
        .unwrap_or(&[])
        .iter()
        .filter_map(|control| {
            let label = control.field_str("label")?.to_owned();
            let option_list = control
                .field("optionList")
                .and_then(PsbValue::as_list)
                .unwrap_or(&[])
                .iter()
                .filter_map(|option| {
                    Some(SelectorOption {
                        label: option.field_str("label")?.to_owned(),
                        off_value: option.field_f32("offValue")?,
                        on_value: option.field_f32("onValue")?,
                    })
                })
                .collect();
            Some(SelectorControl {
                label,
                enabled: control.field_i64("enabled").unwrap_or(1) != 0,
                option_list,
            })
        })
        .collect()
}

fn parse_clamp_controls(value: Option<&PsbValue>) -> Vec<ClampControl> {
    value
        .and_then(PsbValue::as_list)
        .unwrap_or(&[])
        .iter()
        .filter_map(|control| {
            Some(ClampControl {
                label: control.field_str("label").unwrap_or("").to_owned(),
                enabled: control.field_i64("enabled").unwrap_or(1) != 0,
                kind: control.field_i64("type")?,
                var_lr: control.field_str("var_lr")?.to_owned(),
                var_ud: control.field_str("var_ud")?.to_owned(),
                min: control.field_f32("min")?,
                max: control.field_f32("max")?,
            })
        })
        .collect()
}

fn parse_loop_controls(value: Option<&PsbValue>) -> Vec<LoopControl> {
    value
        .and_then(PsbValue::as_list)
        .unwrap_or(&[])
        .iter()
        .map(|control| {
            let transition_list = control
                .field("transitionList")
                .and_then(PsbValue::as_list)
                .unwrap_or(&[])
                .iter()
                .filter_map(parse_loop_transition)
                .collect();
            LoopControl {
                label: control.field_str("label").map(str::to_owned),
                enabled: control.field_i64("enabled").unwrap_or(1) != 0,
                var_loop: control.field_str("var_loop").map(str::to_owned),
                transition_list,
            }
        })
        .collect()
}

fn parse_loop_transition(value: &PsbValue) -> Option<LoopTransition> {
    if let Some(items) = value.as_list() {
        return Some(LoopTransition {
            start: items.first().and_then(PsbValue::as_f32)?,
            end: items.get(1).and_then(PsbValue::as_f32)?,
            duration_ticks: items.get(2).and_then(PsbValue::as_f32)?.max(0.0),
        });
    }
    Some(LoopTransition {
        start: value.field_f32("start")?,
        end: value.field_f32("end")?,
        duration_ticks: value
            .field_f32("duration")
            .or_else(|| value.field_f32("duration_ticks"))?
            .max(0.0),
    })
}

fn parse_mirror_control(value: Option<&PsbValue>) -> Option<MirrorControl> {
    let value = value?;
    let variable_match_list = value
        .field("variableMatchList")
        .and_then(PsbValue::as_list)
        .unwrap_or(&[])
        .iter()
        .filter_map(PsbValue::as_str)
        .map(str::to_owned)
        .collect();
    Some(MirrorControl {
        variable_match_list,
    })
}

fn parse_transition_controls(value: Option<&PsbValue>) -> Vec<TransitionControl> {
    value
        .and_then(PsbValue::as_list)
        .unwrap_or(&[])
        .iter()
        .filter_map(|control| {
            Some(TransitionControl {
                label: control.field_str("label")?.to_owned(),
                enabled: control.field_i64("enabled").unwrap_or(1) != 0,
                fade: control.field_f32("fade"),
                diff: control.field_f32("diff"),
            })
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhysicsControlKind {
    Bust,
    Hair,
    Parts,
}

fn parse_physics_controls(
    value: Option<&PsbValue>,
    kind: PhysicsControlKind,
) -> Vec<PhysicsControl> {
    value
        .and_then(PsbValue::as_list)
        .unwrap_or(&[])
        .iter()
        .filter_map(|control| {
            let definition = PhysicsControlDefinition {
                label: control.field_str("label").unwrap_or("").to_owned(),
                enabled: control.field_i64("enabled").unwrap_or(1) != 0,
                base_layer: control.field_str("baseLayer").map(str::to_owned),
                parameter: control.field_str("parameter").map(str::to_owned),
                var_lr: control.field_str("var_lr").map(str::to_owned),
                var_ud: control.field_str("var_ud").map(str::to_owned),
                var_lrm: control.field_str("var_lrm").map(str::to_owned),
                fields: object_to_map(control)?,
            };
            Some(match kind {
                PhysicsControlKind::Bust => PhysicsControl::Bust(definition),
                PhysicsControlKind::Hair => PhysicsControl::Hair(definition),
                PhysicsControlKind::Parts => PhysicsControl::Parts(definition),
            })
        })
        .collect()
}

fn parse_opaque_controls(value: Option<&PsbValue>) -> Vec<OpaqueControl> {
    value
        .and_then(PsbValue::as_list)
        .unwrap_or(&[])
        .iter()
        .filter_map(|control| {
            Some(OpaqueControl {
                label: control.field_str("label").map(str::to_owned),
                enabled: control.field_i64("enabled").unwrap_or(1) != 0,
                fields: object_to_map(control)?,
            })
        })
        .collect()
}

fn object_to_map(value: &PsbValue) -> Option<BTreeMap<String, PsbValue>> {
    Some(value.as_object()?.iter().cloned().collect())
}

/// Initialize bust physics states from the pipeline's bust controls.
/// Initial bob position from param.p, velocity from param.pv.
/// param.op and param.p are stored in the baseLayer-local coordinate system.
/// The first physics tick translates the saved bob into world space using the
/// current baseLayer marker.
fn init_bust_states(pipeline: &EmoteRuntimePipeline) -> Vec<BustPhysicsState> {
    let mut out = Vec::new();
    for control in &pipeline.physics_controls {
        let PhysicsControl::Bust(def) = control else {
            continue;
        };
        let param = def.fields.get("param");
        let bob = parse_vec3_field(param, "p");
        let root_offset = parse_vec3_field(param, "op");
        let vel = parse_vec3_field(param, "pv");
        let param_ofs = param.and_then(|v| v.field_f32("ofs")).unwrap_or(0.0);
        out.push(BustPhysicsState {
            bob,
            vel,
            ofs: param_ofs,
            first_tick: true,
            root_offset,
            last_anchor: None,
        });
    }
    out
}

/// Initialize pendulum states from the original constructor semantics.
///
/// EPPendControl does not initialize its bobs from param.p/pv in the PSB path.
/// The constructor creates rest0 = root + length[0] * (0, 1, 0) and
/// rest1 = rest0 + length[1] * (0, 1, 0), then copies those rest points into
/// the two current bobs and zeroes both velocities.
fn init_hair_states(pipeline: &EmoteRuntimePipeline) -> Vec<HairPhysicsState> {
    let mut out = Vec::new();
    for control in &pipeline.physics_controls {
        let def = match control {
            PhysicsControl::Hair(def) | PhysicsControl::Parts(def) => def,
            _ => continue,
        };
        let lengths = physics_field_f32_list_2(def, "length");
        let root = [0.0, 0.0, 0.0];
        let rest0 = [root[0], root[1] + lengths[0], root[2]];
        let rest1 = [rest0[0], rest0[1] + lengths[1], rest0[2]];
        let param = def.fields.get("param");
        let param_ofs = param.and_then(|v| v.field_f32("ofs")).unwrap_or(0.0);
        out.push(HairPhysicsState {
            bob: [rest0, rest1],
            vel: [[0.0; 3], [0.0; 3]],
            ofs: param_ofs,
            first_tick: true,
            root_offset: [0.0, 0.0, 0.0],
            last_anchor: None,
            bend_phase: 0.0,
            bend_power: 0.0,
        });
    }
    out
}

fn parse_vec3_field(parent: Option<&PsbValue>, key: &str) -> [f32; 3] {
    let Some(v) = parent.and_then(|p| p.field(key)) else {
        return [0.0; 3];
    };
    [
        v.field_f32("x").unwrap_or(0.0),
        v.field_f32("y").unwrap_or(0.0),
        v.field_f32("z").unwrap_or(0.0),
    ]
}

fn evaluate_selector_control(
    control: &SelectorControl,
    variables: &mut BTreeMap<String, EmoteVariableState>,
) {
    if !control.enabled {
        return;
    }
    let selected = variables
        .get(&control.label)
        .map(|state| state.value.round() as isize)
        .unwrap_or(0);
    for (index, option) in control.option_list.iter().enumerate() {
        let value = if selected == index as isize {
            option.on_value
        } else {
            option.off_value
        };
        set_evaluated_variable(variables, &option.label, value);
    }
}

fn evaluate_clamp_control(
    control: &ClampControl,
    variables: &mut BTreeMap<String, EmoteVariableState>,
) {
    if !control.enabled {
        return;
    }
    let lr = variables
        .get(&control.var_lr)
        .map(|state| state.value)
        .unwrap_or(0.0);
    let ud = variables
        .get(&control.var_ud)
        .map(|state| state.value)
        .unwrap_or(0.0);
    let (next_lr, next_ud) = if control.kind == 0 {
        (
            lr.clamp(control.min, control.max),
            ud.clamp(control.min, control.max),
        )
    } else {
        let radius = lr.hypot(ud);
        let max_radius = control.max.abs().max(f32::EPSILON);
        if radius > max_radius {
            let scale = max_radius / radius;
            (lr * scale, ud * scale)
        } else {
            (lr, ud)
        }
    };
    set_evaluated_variable(variables, &control.var_lr, next_lr);
    set_evaluated_variable(variables, &control.var_ud, next_ud);
}

fn evaluate_loop_control(
    control: &LoopControl,
    elapsed_ticks: f32,
    variables: &mut BTreeMap<String, EmoteVariableState>,
) {
    if !control.enabled {
        return;
    }
    let Some(var_loop) = control.var_loop.as_deref() else {
        return;
    };
    let total: f32 = control
        .transition_list
        .iter()
        .map(|item| item.duration_ticks.max(0.0))
        .sum();
    if total <= f32::EPSILON {
        return;
    }
    let mut t = elapsed_ticks.rem_euclid(total);
    for item in &control.transition_list {
        let duration = item.duration_ticks.max(0.0);
        if t <= duration || duration <= f32::EPSILON {
            let ratio = if duration <= f32::EPSILON {
                1.0
            } else {
                (t / duration).clamp(0.0, 1.0)
            };
            set_evaluated_variable(
                variables,
                var_loop,
                item.start + (item.end - item.start) * ratio,
            );
            return;
        }
        t -= duration;
    }
}

fn evaluate_mirror_control(
    control: &MirrorControl,
    variables: &mut BTreeMap<String, EmoteVariableState>,
) {
    if control.variable_match_list.is_empty() {
        return;
    }
    // The original mirrorControl stores variable match/copy entries. In the PSB
    // text form currently parsed here we may only have flattened labels, so keep
    // the evaluator non-destructive instead of guessing source/target pairing.
    #[cfg(debug_assertions)]
    if !MIRROR_WARNING_PRINTED.swap(true, Ordering::Relaxed) {
        eprintln!("mirrorControl variableMatchList parsed; exact source/target pair layout still needs model-specific decoding: {} entries", control.variable_match_list.len());
    }
    let _ = variables;
}

fn evaluate_transition_control(
    _control: &TransitionControl,
    _variables: &mut BTreeMap<String, EmoteVariableState>,
) {
}

/// EPBustControl group update.
///
/// This follows sub_10273DA0: the baseLayer target is interpolated from the
/// previous frame to the current frame and EPBustControl::step is called over
/// fixed substeps.  The first frame only initializes the root offset.
fn step_bust_physics(
    state: &mut BustPhysicsState,
    def: &PhysicsControlDefinition,
    delta_ticks: f32,
    target_anchor: [f32; 3],
    angle_radians: f32,
    outer_force: [f32; 2],
    output_scale: f32,
    variables: &mut BTreeMap<String, EmoteVariableState>,
) {
    if !def.enabled {
        return;
    }
    if state.first_tick {
        state.bob = [
            target_anchor[0] + state.bob[0],
            target_anchor[1] + state.bob[1],
            target_anchor[2] + state.bob[2],
        ];
        state.last_anchor = Some(target_anchor);
        state.first_tick = false;
        step_bust_physics_once(
            state,
            def,
            0.0,
            target_anchor,
            angle_radians,
            outer_force,
            output_scale,
            variables,
        );
        return;
    }
    if delta_ticks <= PHYSICS_EPSILON_TICKS {
        return;
    }
    let previous_anchor = state.last_anchor.unwrap_or(target_anchor);
    let mut elapsed = 0.0;
    while (delta_ticks - PHYSICS_EPSILON_TICKS) > elapsed {
        let step = (delta_ticks - elapsed).min(PHYSICS_MAX_SUBSTEP_TICKS);
        elapsed += step;
        let ratio = (elapsed / delta_ticks).clamp(0.0, 1.0);
        let anchor = lerp_vec3(previous_anchor, target_anchor, ratio);
        step_bust_physics_once(
            state,
            def,
            step,
            anchor,
            angle_radians,
            outer_force,
            output_scale,
            variables,
        );
    }
    state.last_anchor = Some(target_anchor);
}

fn step_bust_physics_once(
    state: &mut BustPhysicsState,
    def: &PhysicsControlDefinition,
    dt: f32,
    target_anchor: [f32; 3],
    angle_radians: f32,
    outer_force: [f32; 2],
    output_scale: f32,
    variables: &mut BTreeMap<String, EmoteVariableState>,
) {
    let gravity = physics_field_f32(def, "gravity", 0.0);
    let spring = physics_field_f32(def, "spring", 0.0);
    let friction = physics_field_f32(def, "friction", 0.0);
    let scale_x = physics_field_f32(def, "scale_x", 1.0) * output_scale;
    let scale_y = physics_field_f32(def, "scale_y", 1.0) * output_scale;

    let root = [
        target_anchor[0] + state.root_offset[0],
        target_anchor[1] + state.root_offset[1],
        target_anchor[2] + state.root_offset[2],
    ];

    let down = rotated_down(angle_radians);
    let disp = sub_vec3(root, state.bob);

    state.vel[0] += (spring * disp[0] + gravity * down[0] + outer_force[0]) * dt;
    state.vel[1] += (spring * disp[1] + gravity * down[1] + outer_force[1]) * dt;
    state.vel[2] += spring * disp[2] * dt;

    let damp = (1.0 - friction * dt).max(0.0);
    state.vel[0] *= damp;
    state.vel[1] *= damp;
    state.vel[2] *= damp;

    state.bob[0] += state.vel[0] * dt;
    state.bob[1] += state.vel[1] * dt;
    state.bob[2] += state.vel[2] * dt;

    let out_disp = sub_vec3(root, state.bob);
    let mut var_lr = -out_disp[0] * scale_x;
    let mut var_ud = (out_disp[1] + state.ofs) * scale_y;
    deadzone_pair(&mut var_lr, &mut var_ud);

    if let Some(name) = &def.var_lr {
        set_evaluated_variable(variables, name, var_lr);
    }
    if let Some(name) = &def.var_ud {
        set_evaluated_variable(variables, name, var_ud);
    }
}

/// EPPendControl group update for hairControl and partsControl.
///
/// This follows sub_10274A80: interpolate the baseLayer target over fixed
/// substeps, call EPPendControl::step, then apply the bend post-process.
fn step_hair_physics(
    state: &mut HairPhysicsState,
    def: &PhysicsControlDefinition,
    delta_ticks: f32,
    target_anchor: [f32; 3],
    angle_radians: f32,
    outer_force: [f32; 2],
    output_scale: f32,
    variables: &mut BTreeMap<String, EmoteVariableState>,
) {
    if !def.enabled {
        return;
    }
    if state.first_tick {
        let lengths = physics_field_f32_list_2(def, "length");
        let root = [
            target_anchor[0] + state.root_offset[0],
            target_anchor[1] + state.root_offset[1],
            target_anchor[2] + state.root_offset[2],
        ];
        let rest0 = [root[0], root[1] + lengths[0], root[2]];
        let rest1 = [rest0[0], rest0[1] + lengths[1], rest0[2]];
        state.root_offset = [
            root[0] - target_anchor[0],
            root[1] - target_anchor[1],
            root[2] - target_anchor[2],
        ];
        state.bob = [rest0, rest1];
        state.vel = [[0.0; 3], [0.0; 3]];
        state.last_anchor = Some(target_anchor);
        state.first_tick = false;
        step_hair_physics_once(
            state,
            def,
            0.0,
            target_anchor,
            angle_radians,
            outer_force,
            output_scale,
            variables,
        );
        return;
    }
    if delta_ticks <= PHYSICS_EPSILON_TICKS {
        return;
    }
    let previous_anchor = state.last_anchor.unwrap_or(target_anchor);
    let mut elapsed = 0.0;
    while (delta_ticks - PHYSICS_EPSILON_TICKS) > elapsed {
        let step = (delta_ticks - elapsed).min(PHYSICS_MAX_SUBSTEP_TICKS);
        elapsed += step;
        let ratio = (elapsed / delta_ticks).clamp(0.0, 1.0);
        let anchor = lerp_vec3(previous_anchor, target_anchor, ratio);
        step_hair_physics_once(
            state,
            def,
            step,
            anchor,
            angle_radians,
            outer_force,
            output_scale,
            variables,
        );
    }
    state.last_anchor = Some(target_anchor);
}

fn step_hair_physics_once(
    state: &mut HairPhysicsState,
    def: &PhysicsControlDefinition,
    dt: f32,
    target_anchor: [f32; 3],
    angle_radians: f32,
    outer_force: [f32; 2],
    output_scale: f32,
    variables: &mut BTreeMap<String, EmoteVariableState>,
) {
    let gravity = physics_field_f32(def, "gravity", 0.0);
    let friction_x = physics_field_f32(def, "friction_x", 0.0);
    let friction_y = physics_field_f32(def, "friction_y", 0.0);
    let b_rate = physics_field_f32(def, "b_rate", 0.0);
    let v_bound = physics_field_f32(def, "v_bound", 0.0);
    let bend_spd = physics_field_f32(def, "bend_spd", 0.0);
    let bend_vol = physics_field_f32(def, "bend_vol", 0.0);
    let lengths = physics_field_f32_list_2(def, "length");
    let scale_x = physics_field_f32_list_2(def, "scale_x");
    let scale_y = physics_field_f32_list_2(def, "scale_y");
    let ud_eft = physics_field_i64(def, "ud_eft", 0).clamp(0, 1) as usize;

    let root = [
        target_anchor[0] + state.root_offset[0],
        target_anchor[1] + state.root_offset[1],
        target_anchor[2] + state.root_offset[2],
    ];
    let rest0 = [root[0], root[1] + lengths[0], root[2]];
    let rest1 = [rest0[0], rest0[1] + lengths[1], rest0[2]];
    let rest = [rest0, rest1];
    let down = rotated_down(angle_radians);
    let gravity_vec = [
        down[0] * gravity + outer_force[0],
        down[1] * gravity + outer_force[1],
        0.0,
    ];

    for j in 0..2 {
        let hinge = if j == 0 { root } else { state.bob[0] };
        let hinge_to_bob = sub_vec3(state.bob[j], hinge);
        let dist = vec3_len(hinge_to_bob);
        let seg_len = lengths[j].max(f32::EPSILON);
        if dist > seg_len && dist > f32::EPSILON {
            let outward = scale_vec3(hinge_to_bob, 1.0 / dist);
            let inward = scale_vec3(outward, -1.0);
            let excess = dist - seg_len;
            if excess > 0.015625 {
                if j == 1 {
                    state.bob[j] = add_vec3(state.bob[j], scale_vec3(inward, excess));
                    let radial_velocity = dot_vec3(state.vel[j], inward);
                    state.vel[j] = add_vec3(
                        state.vel[j],
                        scale_vec3(inward, -radial_velocity * v_bound * dt),
                    );
                } else {
                    state.vel[j] = add_vec3(state.vel[j], scale_vec3(inward, excess * b_rate * dt));
                }
            }
        }

        state.vel[j][0] += gravity_vec[0] * dt;
        state.vel[j][1] += gravity_vec[1] * dt;
        state.vel[j][2] += gravity_vec[2] * dt;

        state.vel[j][0] -= state.vel[j][0] * friction_x * dt;
        state.vel[j][1] -= state.vel[j][1] * friction_y * dt;
        state.vel[j][2] -= state.vel[j][2] * friction_y * dt;

        state.bob[j] = add_vec3(state.bob[j], scale_vec3(state.vel[j], dt));
    }

    let d0 = sub_vec3(rest[0], state.bob[0]);
    let d1 = sub_vec3(rest[1], state.bob[1]);
    let mut var_lr = -d0[0] * scale_x[0] * output_scale;
    let mut var_lrm = -d1[0] * scale_x[1] * output_scale;
    let ud_disp = if ud_eft == 0 { d0[1] } else { d1[1] };
    let mut var_ud = (state.ofs - ud_disp) * scale_y[ud_eft] * output_scale;

    apply_pend_bend(state, bend_spd, bend_vol, dt, &mut var_lr, &mut var_lrm);
    deadzone_triple(&mut var_lr, &mut var_lrm, &mut var_ud);

    if let Some(name) = &def.var_lr {
        set_evaluated_variable(variables, name, var_lr);
    }
    if let Some(name) = &def.var_lrm {
        set_evaluated_variable(variables, name, var_lrm);
    }
    if let Some(name) = &def.var_ud {
        set_evaluated_variable(variables, name, var_ud);
    }
}

fn apply_pend_bend(
    state: &mut HairPhysicsState,
    bend_spd: f32,
    bend_vol: f32,
    dt: f32,
    var_lr: &mut f32,
    var_lrm: &mut f32,
) {
    if bend_spd == 0.0 || bend_vol == 0.0 {
        return;
    }
    let trigger = var_lr.abs();
    if trigger <= PEND_BEND_TRIGGER_VALUE {
        state.bend_power = (state.bend_power - PEND_BEND_POWER_STEP * dt).max(0.0);
    } else {
        state.bend_power = (state.bend_power + PEND_BEND_POWER_STEP * dt).min(1.0);
    }
    state.bend_phase = (state.bend_phase + bend_spd * state.bend_power * dt).rem_euclid(TAU);
    let bend = state.bend_phase.sin() * state.bend_power * bend_vol;
    *var_lrm += bend;
    *var_lr -= bend;
}

fn rotated_down(angle_radians: f32) -> [f32; 3] {
    let c = (-angle_radians).cos();
    let s = (-angle_radians).sin();
    [-s, c, 0.0]
}

fn lerp_vec3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn add_vec3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub_vec3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale_vec3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot_vec3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn vec3_len(a: [f32; 3]) -> f32 {
    dot_vec3(a, a).sqrt()
}

fn deadzone_pair(a: &mut f32, b: &mut f32) {
    if a.abs() <= 0.01 && b.abs() <= 0.01 {
        *a = 0.0;
        *b = 0.0;
    }
}

fn deadzone_triple(a: &mut f32, b: &mut f32, c: &mut f32) {
    if a.abs() <= 0.01 && b.abs() <= 0.01 && c.abs() <= 0.01 {
        *a = 0.0;
        *b = 0.0;
        *c = 0.0;
    }
}

fn physics_field_f32(def: &PhysicsControlDefinition, key: &str, default: f32) -> f32 {
    def.fields
        .get("param")
        .and_then(|param| param.field_f32(key))
        .or_else(|| def.fields.get(key).and_then(PsbValue::as_f32))
        .unwrap_or(default)
}

fn physics_field_i64(def: &PhysicsControlDefinition, key: &str, default: i64) -> i64 {
    def.fields
        .get("param")
        .and_then(|param| param.field_i64(key))
        .or_else(|| def.fields.get(key).and_then(PsbValue::as_i64))
        .unwrap_or(default)
}

fn physics_field_f32_list_2(def: &PhysicsControlDefinition, key: &str) -> [f32; 2] {
    if let Some(param) = def.fields.get("param") {
        let parsed = parse_f32_value_list_2(param.field(key));
        if parsed != [0.0, 0.0] {
            return parsed;
        }
    }
    parse_f32_value_list_2(def.fields.get(key))
}

fn parse_f32_value_list_2(value: Option<&PsbValue>) -> [f32; 2] {
    let Some(value) = value else {
        return [0.0, 0.0];
    };
    match value {
        PsbValue::List(items) => {
            let a = items.first().and_then(PsbValue::as_f32).unwrap_or(0.0);
            let b = items.get(1).and_then(PsbValue::as_f32).unwrap_or(a);
            [a, b]
        }
        _ => {
            let v = value.as_f32().unwrap_or(0.0);
            [v, v]
        }
    }
}

fn set_evaluated_variable(
    variables: &mut BTreeMap<String, EmoteVariableState>,
    name: &str,
    value: f32,
) {
    if name.is_empty() || !value.is_finite() {
        return;
    }
    let state = variables
        .entry(name.to_owned())
        .or_insert_with(|| EmoteVariableState {
            info: EmoteVariableInfo {
                name: name.to_owned(),
                default_value: value,
                min_value: None,
                max_value: None,
                frames: Vec::new(),
            },
            value,
            target: None,
        });
    state.value = clamp_variable_value(&state.info, value);
    state.target = None;
}

pub fn collect_emote_timelines(psb: &PsbFile) -> Vec<EmoteTimeline> {
    let mut out = Vec::new();
    let aliases = collect_variable_frame_aliases(psb);
    if let Some(metadata) = psb.root.field("metadata") {
        if let Some(timeline_root) = metadata.field("timelineControl") {
            collect_timeline_nodes(timeline_root, "", false, &aliases, &mut out);
        }
        for key in CONTROL_METADATA_KEYS {
            if let Some(control_root) = metadata.field(key) {
                let path = format!("@control/{key}");
                collect_timeline_nodes(control_root, &path, false, &aliases, &mut out);
            }
        }
    }

    out.sort_by(|a, b| a.name.cmp(&b.name));
    out.dedup_by(|a, b| a.name.as_str() == b.name.as_str());
    out
}

fn collect_timeline_nodes(
    value: &PsbValue,
    path: &str,
    inherited_difference: bool,
    aliases: &BTreeMap<String, BTreeMap<String, f32>>,
    out: &mut Vec<EmoteTimeline>,
) {
    match value {
        PsbValue::List(items) => {
            for (index, item) in items.iter().enumerate() {
                let next_path = join_timeline_path(path, &index.to_string());
                collect_timeline_nodes(item, &next_path, inherited_difference, aliases, out);
            }
        }
        PsbValue::Object(fields) => {
            if value.field_str("type") == Some("folder") {
                let label = value.field_str("label").unwrap_or("");
                let folder_path = join_timeline_path(path, label);
                if let Some(children) = value.field("children") {
                    collect_timeline_nodes(
                        children,
                        &folder_path,
                        psb_field_bool_like(value, "diff").unwrap_or(inherited_difference),
                        aliases,
                        out,
                    );
                }
                return;
            }

            if let Some(variable_list) = value.field("variableList").and_then(PsbValue::as_list) {
                let label = value
                    .field_str("label")
                    .or_else(|| value.field_str("name"))
                    .or_else(|| value.field_str("id"))
                    .unwrap_or("timeline");
                let hinted_path = value
                    .field_str("path_hint")
                    .or_else(|| value.field_str("hint_path"))
                    .map(str::to_owned);
                let full_name = hinted_path
                    .clone()
                    .unwrap_or_else(|| join_timeline_path(path, label));
                let name = if full_name.is_empty() {
                    label.to_owned()
                } else {
                    full_name
                };

                let mut variables = Vec::new();
                let mut duration = value
                    .field_f32("lastTime")
                    .or_else(|| value.field_f32("loopTime"))
                    .unwrap_or(0.0)
                    .max(0.0);

                for variable in variable_list {
                    let Some(var_name) = variable_object_name(variable) else {
                        continue;
                    };
                    let Some(frame_list) = variable.field("frameList").and_then(PsbValue::as_list)
                    else {
                        continue;
                    };
                    let mut frames = Vec::new();
                    for frame in frame_list {
                        let time = frame.field_f32("time").unwrap_or(0.0).max(0.0);
                        let Some(content) = frame.field("content") else {
                            continue;
                        };
                        let value =
                            timeline_content_value(&var_name, content, aliases).unwrap_or(0.0);
                        let easing = content
                            .field_f32("easing")
                            .or_else(|| frame.field_f32("easing"))
                            .unwrap_or(0.0);
                        if value.is_finite() {
                            duration = duration.max(time);
                            frames.push(EmoteTimelineFrame {
                                time_ticks: time,
                                value,
                                easing,
                            });
                        }
                    }
                    frames.sort_by(|a, b| a.time_ticks.total_cmp(&b.time_ticks));
                    if !frames.is_empty() {
                        variables.push(EmoteTimelineVariable {
                            name: var_name,
                            frames,
                        });
                    }
                }

                if !variables.is_empty() {
                    out.push(EmoteTimeline {
                        name,
                        path: hinted_path,
                        duration_ticks: duration,
                        variables,
                        is_difference: psb_field_bool_like(value, "diff")
                            .unwrap_or(inherited_difference),
                    });
                }
            }

            for (key, child) in fields {
                if matches!(
                    key.as_str(),
                    "variableList" | "frameList" | "content" | "variable"
                ) {
                    continue;
                }
                let label = child
                    .field_str("label")
                    .or_else(|| child.field_str("name"))
                    .or_else(|| child.field_str("id"))
                    .unwrap_or(key);
                let next_path = join_timeline_path(path, label);
                collect_timeline_nodes(
                    child,
                    &next_path,
                    psb_field_bool_like(child, "diff").unwrap_or(inherited_difference),
                    aliases,
                    out,
                );
            }
        }
        _ => {}
    }
}

fn timeline_content_value(
    variable_name: &str,
    content: &PsbValue,
    aliases: &BTreeMap<String, BTreeMap<String, f32>>,
) -> Option<f32> {
    for key in ["value", "frame", "curValue", "targetValue", variable_name] {
        if let Some(value) = content.field(key) {
            if let Some(n) = value.as_f32() {
                return Some(n);
            }
            if let Some(label) = value.as_str() {
                if let Ok(n) = label.parse::<f32>() {
                    return Some(n);
                }
                if let Some(n) = aliases
                    .get(variable_name)
                    .and_then(|map| map.get(label))
                    .copied()
                {
                    return Some(n);
                }
            }
        }
    }
    if let Some(label) = content.field_str("label") {
        if let Some(n) = aliases
            .get(variable_name)
            .and_then(|map| map.get(label))
            .copied()
        {
            return Some(n);
        }
    }
    None
}

fn collect_variable_frame_aliases(psb: &PsbFile) -> BTreeMap<String, BTreeMap<String, f32>> {
    let mut out = BTreeMap::<String, BTreeMap<String, f32>>::new();
    if let Some(metadata) = psb.root.field("metadata") {
        collect_variable_frame_alias_nodes(metadata.field("variableList"), &mut out);
        collect_variable_frame_alias_nodes(metadata.field("instantVariableList"), &mut out);
        collect_variable_frame_alias_nodes(metadata.field("physicsVariableList"), &mut out);
        if let Some(timeline_control) = metadata.field("timelineControl") {
            collect_variable_frame_alias_nodes(timeline_control.field("variableList"), &mut out);
        }
        for key in CONTROL_METADATA_KEYS {
            if let Some(control) = metadata.field(key) {
                collect_control_variable_aliases(control, &mut out);
            }
        }
    }
    out
}

fn collect_variable_frame_alias_nodes(
    value: Option<&PsbValue>,
    out: &mut BTreeMap<String, BTreeMap<String, f32>>,
) {
    let Some(value) = value else {
        return;
    };
    match value {
        PsbValue::List(items) => {
            for item in items {
                collect_one_variable_frame_alias(item, out);
            }
        }
        PsbValue::Object(fields) => {
            for (_key, child) in fields {
                collect_one_variable_frame_alias(child, out);
            }
        }
        _ => {}
    }
}

fn collect_control_variable_aliases(
    value: &PsbValue,
    out: &mut BTreeMap<String, BTreeMap<String, f32>>,
) {
    collect_one_variable_frame_alias(value, out);
    collect_variable_frame_alias_nodes(value.field("variableList"), out);
    collect_variable_frame_alias_nodes(value.field("variables"), out);
    if let Some(variable) = value.field("variable") {
        collect_one_variable_frame_alias(variable, out);
    }

    match value {
        PsbValue::List(items) => {
            for item in items {
                collect_control_variable_aliases(item, out);
            }
        }
        PsbValue::Object(fields) => {
            for (key, child) in fields {
                if matches!(
                    key.as_str(),
                    "variableList" | "variables" | "variable" | "frameList" | "content"
                ) {
                    continue;
                }
                collect_control_variable_aliases(child, out);
            }
        }
        _ => {}
    }
}

fn collect_one_variable_frame_alias(
    variable: &PsbValue,
    out: &mut BTreeMap<String, BTreeMap<String, f32>>,
) {
    let Some(name) = variable_object_name(variable) else {
        return;
    };
    let Some(frame_list) = variable.field("frameList").and_then(PsbValue::as_list) else {
        return;
    };
    let map = out.entry(name).or_default();
    for frame in frame_list {
        let label = frame.field_str("label").or_else(|| {
            frame
                .field("content")
                .and_then(|content| content.field_str("label"))
        });
        let value = frame
            .field_f32("frame")
            .or_else(|| {
                frame
                    .field("content")
                    .and_then(|content| content.field_f32("frame"))
            })
            .or_else(|| frame.field_f32("value"))
            .or_else(|| {
                frame
                    .field("content")
                    .and_then(|content| content.field_f32("value"))
            });
        if let (Some(label), Some(value)) = (label, value) {
            if !label.is_empty() && value.is_finite() {
                map.insert(label.to_owned(), value);
            }
        }
    }
}

fn join_timeline_path(path: &str, label: &str) -> String {
    match (path.is_empty(), label.is_empty()) {
        (true, true) => String::new(),
        (true, false) => label.to_owned(),
        (false, true) => path.to_owned(),
        (false, false) => format!("{path}/{label}"),
    }
}

pub fn collect_emote_variables(psb: &PsbFile) -> Vec<EmoteVariableInfo> {
    let mut out = BTreeMap::<String, EmoteVariableInfo>::new();

    if let Some(metadata) = psb.root.field("metadata") {
        collect_variable_list(metadata.field("variableList"), &mut out);
        collect_variable_list(metadata.field("instantVariableList"), &mut out);
        collect_variable_list(metadata.field("physicsVariableList"), &mut out);
        if let Some(timeline_control) = metadata.field("timelineControl") {
            collect_variable_list(timeline_control.field("variableList"), &mut out);
            collect_control_variables(timeline_control, &mut out);
        }
        for key in CONTROL_METADATA_KEYS {
            if let Some(control) = metadata.field(key) {
                collect_control_variables(control, &mut out);
            }
        }
    }

    collect_variable_list(psb.root.field("parameter"), &mut out);
    collect_variable_list(psb.root.field("parameters"), &mut out);
    collect_parameter_variables_recursive(&psb.root, &mut out);
    collect_mesh_combinator_variables(&psb.root, &mut out);
    for timeline in collect_emote_timelines(psb) {
        for variable in &timeline.variables {
            merge_timeline_variable_info(variable, &mut out);
        }
    }

    out.into_values().collect()
}

fn collect_parameter_variables_recursive(
    value: &PsbValue,
    out: &mut BTreeMap<String, EmoteVariableInfo>,
) {
    if let Some(parameter_list) = value.field("parameter") {
        collect_variable_list(Some(parameter_list), out);
    }
    match value {
        PsbValue::List(values) => {
            for child in values {
                collect_parameter_variables_recursive(child, out);
            }
        }
        PsbValue::Object(fields) => {
            for (key, child) in fields {
                if key == "parameter" {
                    continue;
                }
                collect_parameter_variables_recursive(child, out);
            }
        }
        _ => {}
    }
}

fn collect_control_variables(value: &PsbValue, out: &mut BTreeMap<String, EmoteVariableInfo>) {
    collect_one_variable(value, out);
    collect_variable_list(value.field("variableList"), out);
    collect_variable_list(value.field("variables"), out);
    if let Some(variable) = value.field("variable") {
        collect_one_variable(variable, out);
    }

    match value {
        PsbValue::List(values) => {
            for child in values {
                collect_control_variables(child, out);
            }
        }
        PsbValue::Object(fields) => {
            for (key, child) in fields {
                if matches!(
                    key.as_str(),
                    "variableList" | "variables" | "variable" | "frameList" | "content"
                ) {
                    continue;
                }
                collect_control_variables(child, out);
            }
        }
        _ => {}
    }
}

fn collect_mesh_combinator_variables(
    value: &PsbValue,
    out: &mut BTreeMap<String, EmoteVariableInfo>,
) {
    if let Some(combinators) = value
        .field("meshCombinator")
        .and_then(|mesh_combinator| mesh_combinator.field("combinatorList"))
        .and_then(PsbValue::as_list)
    {
        for combinator in combinators {
            if let Some(variable) = combinator.field("variable") {
                collect_one_variable(variable, out);
            }
        }
    }

    match value {
        PsbValue::List(values) => {
            for child in values {
                collect_mesh_combinator_variables(child, out);
            }
        }
        PsbValue::Object(fields) => {
            for (_key, child) in fields {
                collect_mesh_combinator_variables(child, out);
            }
        }
        _ => {}
    }
}

fn collect_variable_list(value: Option<&PsbValue>, out: &mut BTreeMap<String, EmoteVariableInfo>) {
    let Some(value) = value else {
        return;
    };

    match value {
        PsbValue::List(values) => {
            for child in values {
                collect_one_variable(child, out);
            }
        }
        PsbValue::Object(fields) => {
            for (_key, child) in fields {
                collect_one_variable(child, out);
            }
        }
        _ => {}
    }
}

fn collect_one_variable(value: &PsbValue, out: &mut BTreeMap<String, EmoteVariableInfo>) {
    let Some(name) = variable_object_name(value) else {
        return;
    };

    let frames = variable_frame_infos(value);
    let frame_min = frames.iter().map(|frame| frame.value).reduce(f32::min);
    let frame_max = frames.iter().map(|frame| frame.value).reduce(f32::max);

    let next = EmoteVariableInfo {
        name: name.clone(),
        default_value: value
            .field_f32("default")
            .or_else(|| value.field_f32("defaultValue"))
            .or_else(|| value.field_f32("initial"))
            .or_else(|| value.field_f32("curValue"))
            .or_else(|| value.field_f32("value"))
            .or_else(|| frames.first().map(|frame| frame.value))
            .unwrap_or(0.0),
        min_value: value
            .field_f32("min")
            .or_else(|| value.field_f32("minValue"))
            .or_else(|| value.field_f32("rangeBegin"))
            .or(frame_min),
        max_value: value
            .field_f32("max")
            .or_else(|| value.field_f32("maxValue"))
            .or_else(|| value.field_f32("rangeEnd"))
            .or(frame_max),
        frames,
    };

    merge_variable_info(name, next, out);
}

fn merge_timeline_variable_info(
    variable: &EmoteTimelineVariable,
    out: &mut BTreeMap<String, EmoteVariableInfo>,
) {
    if variable.name.is_empty() {
        return;
    }
    let mut frames = Vec::new();
    for frame in &variable.frames {
        frames.push(EmoteVariableFrameInfo {
            label: format!("{:.3}", frame.time_ticks),
            value: frame.value,
        });
    }
    let frame_min = frames.iter().map(|frame| frame.value).reduce(f32::min);
    let frame_max = frames.iter().map(|frame| frame.value).reduce(f32::max);
    let default_value = variable
        .frames
        .first()
        .map(|frame| frame.value)
        .unwrap_or(0.0);
    merge_variable_info(
        variable.name.clone(),
        EmoteVariableInfo {
            name: variable.name.clone(),
            default_value,
            min_value: frame_min,
            max_value: frame_max,
            frames,
        },
        out,
    );
}

fn merge_variable_info(
    name: String,
    next: EmoteVariableInfo,
    out: &mut BTreeMap<String, EmoteVariableInfo>,
) {
    let entry = out.entry(name).or_insert_with(|| next.clone());
    if entry.frames.is_empty() && !next.frames.is_empty() {
        entry.default_value = next.default_value;
    }
    if let Some(min) = next.min_value {
        merge_range_value(&mut entry.min_value, &mut entry.max_value, min);
    }
    if let Some(max) = next.max_value {
        merge_range_value(&mut entry.min_value, &mut entry.max_value, max);
    }
    for frame in next.frames {
        if !entry.frames.iter().any(|existing| {
            existing.label == frame.label && (existing.value - frame.value).abs() <= f32::EPSILON
        }) {
            merge_range_value(&mut entry.min_value, &mut entry.max_value, frame.value);
            entry.frames.push(frame);
        }
    }
}

fn variable_frame_infos(value: &PsbValue) -> Vec<EmoteVariableFrameInfo> {
    let Some(frame_list) = value.field("frameList").and_then(PsbValue::as_list) else {
        return Vec::new();
    };
    let mut frames = Vec::new();
    for frame in frame_list {
        let label = frame
            .field_str("label")
            .or_else(|| {
                frame
                    .field("content")
                    .and_then(|content| content.field_str("label"))
            })
            .unwrap_or("");
        let value = frame
            .field_f32("frame")
            .or_else(|| {
                frame
                    .field("content")
                    .and_then(|content| content.field_f32("frame"))
            })
            .or_else(|| frame.field_f32("value"))
            .or_else(|| {
                frame
                    .field("content")
                    .and_then(|content| content.field_f32("value"))
            });
        if let Some(value) = value.filter(|v| v.is_finite()) {
            frames.push(EmoteVariableFrameInfo {
                label: label.to_owned(),
                value,
            });
        }
    }
    frames.sort_by(|a, b| {
        a.value
            .total_cmp(&b.value)
            .then_with(|| a.label.cmp(&b.label))
    });
    frames.dedup_by(|a, b| a.label == b.label && (a.value - b.value).abs() <= f32::EPSILON);
    frames
}

fn psb_field_bool_like(value: &PsbValue, name: &str) -> Option<bool> {
    match value.field(name)? {
        PsbValue::Bool(v) => Some(*v),
        PsbValue::Int(v) => Some(*v != 0),
        PsbValue::Float(v) => Some(*v != 0.0),
        PsbValue::Double(v) => Some(*v != 0.0),
        PsbValue::String(v) => match v.as_str() {
            "true" | "TRUE" | "True" | "1" => Some(true),
            "false" | "FALSE" | "False" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn parse_api_log_entry(line: &str) -> Option<EmoteApiLogEntry> {
    let mut parts = line.split('\t');
    let command = parts.next()?.trim();
    if command.is_empty() {
        return None;
    }
    Some(EmoteApiLogEntry::new(
        command.to_owned(),
        parts.map(str::to_owned).collect(),
    ))
}

fn variable_object_name(obj: &PsbValue) -> Option<String> {
    for key in ["id", "key", "name", "label"] {
        if let Some(value) = obj.field_str(key).filter(|s| !s.is_empty()) {
            return Some(value.to_owned());
        }
    }
    None
}
