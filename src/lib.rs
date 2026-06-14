use std::{collections::HashMap, time::Instant};

#[repr(C)]
#[derive(serde::Deserialize, Clone, Debug, PartialEq)]
pub struct Vertex {
    pub pos: Vec2,
    pub uv: Vec2,
    pub init_pos: Vec2,
}

#[derive(serde::Deserialize, Clone, Debug, Default, Copy, PartialEq)]
pub struct Tint {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[derive(serde::Deserialize, Clone, Debug, Default, Copy)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }
}

macro_rules! impl_assign_for_vec2 {
    ($trait:ident, $method:ident, $op:tt) => {
        impl std::ops::$trait for Vec2 {
            fn $method(&mut self, other: Vec2) {
                self.x $op other.x;
                self.y $op other.y;
            }
        }
    };
}

macro_rules! impl_assign_f32_for_vec2 {
    ($trait:ident, $method:ident, $op:tt) => {
        impl std::ops::$trait<f32> for Vec2 {
            fn $method(&mut self, other: f32) {
                self.x $op other;
                self.y $op other;
            }
        }
    };
}

macro_rules! impl_for_vec2 {
    ($trait:ident, $method:ident, $op:tt) => {
        impl std::ops::$trait for Vec2 {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: Self) -> Self {
                Self {
                    x: self.x $op rhs.x,
                    y: self.y $op rhs.y,
                }
            }
        }
    };
}

macro_rules! impl_f32_for_vec2 {
    ($trait:ident, $method:ident, $op:tt) => {
        impl std::ops::$trait<f32> for Vec2 {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: f32) -> Self {
                Self {
                    x: self.x $op rhs,
                    y: self.y $op rhs,
                }
            }
        }
    };
}

impl_assign_for_vec2!(AddAssign, add_assign, +=);
impl_assign_for_vec2!(SubAssign, sub_assign, -=);
impl_assign_for_vec2!(DivAssign, div_assign, /=);
impl_assign_for_vec2!(MulAssign, mul_assign, *=);

impl_assign_f32_for_vec2!(AddAssign, add_assign, +=);
impl_assign_f32_for_vec2!(SubAssign, sub_assign, -=);
impl_assign_f32_for_vec2!(DivAssign, div_assign, /=);
impl_assign_f32_for_vec2!(MulAssign, mul_assign, *=);

impl_for_vec2!(Add, add, +);
impl_for_vec2!(Sub, sub, -);
impl_for_vec2!(Mul, mul, *);
impl_for_vec2!(Div, div, /);

impl_f32_for_vec2!(Add, add, +);
impl_f32_for_vec2!(Sub, sub, -);
impl_f32_for_vec2!(Mul, mul, *);
impl_f32_for_vec2!(Div, div, /);

impl PartialEq for Vec2 {
    fn eq(&self, other: &Vec2) -> bool {
        return self.x == other.x && self.y == other.y;
    }
}
#[derive(serde::Deserialize, Clone, Default, Debug)]
#[serde(default)]
pub struct Animation {
    pub name: String,
    pub fps: u32,
    pub keyframes: Vec<Keyframe>,
}

#[derive(serde::Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct InverseKinematics {
    pub family_id: i32,
    pub constraint: String,
    pub mode: String,
    pub target_id: i32,
    pub bone_ids: Vec<u32>,

    pub init_constraint: String,
    pub init_mode: String,
    pub init_mimic_target: bool,
}

#[derive(serde::Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Visuals {
    pub tex: String,
    pub tint: Tint,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub binds: Vec<BoneBind>,
    pub zindex: i32,

    pub init_tex: String,
    pub init_zindex: i32,
    pub init_tint: Tint,
}

#[derive(serde::Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Physics {
    pub global_pos: Vec2,
    pub pos_damping: f32,
    pub pos_ratio: f32,

    pub global_rot: f32,
    pub global_orbit: f32,
    pub global_orbit_diff: f32,
    pub global_orbit_vel: f32,
    pub rot_damping: f32,
    pub rot_bounce: f32,
    pub rot_vel: f32,
    pub sway: f32,

    pub global_scale: Vec2,
    pub scale_damping: f32,
    pub scale_ratio: f32,
}

#[derive(serde::Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Bone {
    pub id: u32,
    pub name: String,
    pub parent_id: i32,

    pub rot: f32,
    pub scale: Vec2,
    pub pos: Vec2,
    pub hidden: bool,

    pub ik_family_id: i32,
    pub visuals_id: i32,
    pub physics_id: i32,

    pub init_rot: f32,
    pub init_scale: Vec2,
    pub init_pos: Vec2,
    pub init_hidden: bool,
}

#[derive(serde::Deserialize, Clone, PartialEq, Default, Debug)]
#[serde(default)]
pub struct BoneBind {
    pub bone_id: i32,
    pub is_path: bool,
    pub verts: Vec<BoneBindVert>,
}

#[derive(serde::Deserialize, Clone, PartialEq, Default, Debug)]
#[serde(default)]
pub struct BoneBindVert {
    pub id: u32,
    pub weight: f32,
}

#[derive(Eq, Ord, PartialEq, PartialOrd, serde::Deserialize, Clone, Default, Debug)]
pub enum AnimElement {
    #[default]
    PositionX,
    PositionY,
    Rotation,
    ScaleX,
    ScaleY,
    Zindex,
    Texture,
    IkConstraint,
}

#[derive(PartialEq, serde::Deserialize, Clone, Default, Debug)]
#[serde(default)]
pub struct Keyframe {
    pub frame: u32,
    pub bone_id: u32,
    pub element: String,
    pub value: f32,
    pub next_kf: i32,
    pub value_str: String,
    pub start_handle: Vec2,
    pub end_handle: Vec2,
    pub handle_preset: HandlePreset,
    #[serde(skip)]
    pub label_top: f32,
}

#[derive(PartialEq, serde::Deserialize, Clone, Default, Debug)]
pub enum HandlePreset {
    #[default]
    Linear,
    SineIn,
    SineOut,
    SineInOut,
    None,
    Custom,
}

#[derive(serde::Deserialize, Clone, Default, Debug)]
#[serde(default)]
pub struct Style {
    pub id: u32,
    pub name: String,
    pub active: bool,
    pub textures: Vec<Texture>,
}

#[derive(serde::Deserialize, Clone, Copy, Default, PartialEq, Debug)]
pub enum JointConstraint {
    #[default]
    None,
    Clockwise,
    CounterClockwise,
}

#[derive(serde::Deserialize, Copy, Clone, Default, PartialEq, Debug)]
pub enum InverseKinematicsMode {
    #[default]
    FABRIK,
    Arc,
}

#[derive(serde::Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct TexAtlas {
    pub filename: String,
    pub size: Vec2,
}

#[derive(serde::Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Armature {
    pub baked_ik: bool,
    pub bones: Vec<Bone>,
    pub constructed_bones: Vec<Bone>,
    pub animations: Vec<Animation>,
    pub textures: Vec<Texture>,
    pub styles: Vec<Style>,
    pub atlases: Vec<TexAtlas>,
    pub inverse_kinematics: Vec<InverseKinematics>,
    pub visuals: Vec<Visuals>,
    pub physics: Vec<Physics>,
}

#[derive(serde::Deserialize, Clone, Default, Debug, PartialEq)]
#[serde(default)]
pub struct Texture {
    pub offset: Vec2,
    pub size: Vec2,
    pub name: String,
    pub atlas_idx: u32,
}

/// Process bones with animations.
pub fn animate(
    bones: &mut Vec<Bone>,
    inverse_kinematics: &mut Vec<InverseKinematics>,
    visuals: &mut Vec<Visuals>,
    anims: &Vec<&Animation>,
    frames: &Vec<u32>,
    blend_frames: &Vec<u32>,
) {
    // keeps track of animated elements. Bone elements not included will be reset
    let mut reset_map: HashMap<u32, Vec<&str>> = HashMap::new();

    for a in 0..anims.len() {
        for k in 0..anims[a].keyframes.len() {
            let kf = &anims[a].keyframes[k];

            // add this keyframe bone and element, to be used later
            let mut new: Vec<&str> = vec![];
            if let Some(reset) = reset_map.get(&kf.bone_id) {
                new = reset.clone();
            }
            if !new.contains(&kf.element.as_str()) {
                new.push(kf.element.as_str());
                reset_map.insert(kf.bone_id, new);
            }

            // skip animation if current keyframes are beyond this frame
            if kf.frame > frames[a] {
                break;
            }

            // set next_kf to itself, if it's -1
            let mut nkf = kf.next_kf;
            if nkf == -1 {
                nkf = k as i32;
            }

            let next_kf = &anims[a].keyframes[nkf as usize];

            // skip keyframe if it's not the last, and would not be animated
            let is_last = nkf == k as i32;
            let is_before_frame = next_kf.frame < frames[a];
            if is_before_frame && !is_last {
                continue;
            }

            let bone = &mut bones[kf.bone_id as usize];
            let f = frames[a];
            let bf = blend_frames[a];

            // animate basic fields
            match kf.element.as_str() {
                "PositionX" => interpolate_keyframes(&mut bone.pos.x, kf, next_kf, f, bf),
                "PositionY" => interpolate_keyframes(&mut bone.pos.y, kf, next_kf, f, bf),
                "Rotation" => interpolate_keyframes(&mut bone.rot, kf, next_kf, f, bf),
                "ScaleX" => interpolate_keyframes(&mut bone.scale.x, kf, next_kf, f, bf),
                "ScaleY" => interpolate_keyframes(&mut bone.scale.y, kf, next_kf, f, bf),
                "Hidden" => bone.hidden = kf.value == 1.,
                _ => {}
            }

            // animate visual fields
            if let Some(visual) = visuals.get_mut(bone.visuals_id as usize) {
                match kf.element.as_str() {
                    "TintR" => interpolate_keyframes(&mut visual.tint.r, kf, next_kf, f, bf),
                    "TintG" => interpolate_keyframes(&mut visual.tint.g, kf, next_kf, f, bf),
                    "TintB" => interpolate_keyframes(&mut visual.tint.b, kf, next_kf, f, bf),
                    "TintA" => interpolate_keyframes(&mut visual.tint.a, kf, next_kf, f, bf),
                    "Tex" => visual.tex = kf.value_str.clone(),
                    _ => {}
                }
            }

            // animate inverse kinematics fields
            if let Some(ik) = inverse_kinematics.get_mut(bone.ik_family_id as usize) {
                match kf.element.as_str() {
                    "IkConstraint" => ik.constraint = kf.value_str.clone(),
                    _ => {}
                }
            }
        }
    }

    // reset non-animated bone elements
    for bone in bones {
        let mut reset: &Vec<&str> = &vec![];
        if let Some(this_reset) = reset_map.get(&bone.id) {
            reset = this_reset;
        }

        let z = Vec2::new(0., 0.);
        let sf = blend_frames[0];
        let f = frames[0];

        // reset basic fields
        if !reset.contains(&"PositionX") {
            bone.pos.x = interpolate(f, sf, bone.pos.x, bone.init_pos.x, z, z);
        }
        if !reset.contains(&"PositionY") {
            bone.pos.y = interpolate(f, sf, bone.pos.y, bone.init_pos.y, z, z);
        }
        if !reset.contains(&"Rotation") {
            bone.rot = interpolate(f, sf, bone.rot, bone.init_rot, z, z);
        }
        if !reset.contains(&"ScaleX") {
            bone.scale.x = interpolate(f, sf, bone.scale.x, bone.init_scale.x, z, z);
        }
        if !reset.contains(&"ScaleY") {
            bone.scale.y = interpolate(f, sf, bone.scale.y, bone.init_scale.y, z, z);
        }
        if !reset.contains(&"Hidden") {
            bone.hidden = bone.init_hidden;
        }

        // reset visuals data
        if let Some(visual) = visuals.get_mut(bone.visuals_id as usize) {
            if !reset.contains(&"Texture") {
                visual.tex = visual.init_tex.clone();
            }
            if !reset.contains(&"TintR") {
                visual.tint.r = interpolate(f, sf, visual.tint.r, visual.init_tint.r, z, z);
            }
            if !reset.contains(&"TintG") {
                visual.tint.g = interpolate(f, sf, visual.tint.g, visual.init_tint.g, z, z);
            }
            if !reset.contains(&"TintB") {
                visual.tint.b = interpolate(f, sf, visual.tint.b, visual.init_tint.b, z, z);
            }
            if !reset.contains(&"TintA") {
                visual.tint.a = interpolate(f, sf, visual.tint.a, visual.init_tint.a, z, z);
            }
        }

        // reset inverse kinematics data
        if let Some(ik) = inverse_kinematics.get_mut(bone.ik_family_id as usize) {
            if !reset.contains(&"IkConstraint") {
                ik.constraint = ik.init_constraint.clone();
            }
        }
    }
}

pub fn get_bone_texture(bone_tex: String, styles: &Vec<&Style>) -> Option<Texture> {
    for style in styles {
        if let Some(tex) = style.textures.iter().find(|t| t.name == bone_tex) {
            return Some(tex.clone());
        }
    }
    return None;
}

/// Apply child-parent inheritance.
/// Must be run twice, before and after `inverse_kinematics()`.
pub fn inheritance(bones: &mut Vec<Bone>, ik_rots: HashMap<u32, f32>, physics: &Vec<Physics>) {
    for b in 0..bones.len() {
        if bones[b].parent_id != -1 {
            let parent = &bones[bones[b].parent_id as usize];
            let parent_pos = parent.pos;
            let parent_scale = parent.scale;

            let mut orbit_rot = bones[bones[b].parent_id as usize].rot;
            // apply orbital difference, if rotation resistance physics is active
            if let Some(phys) = physics.get(bones[b].physics_id as usize) {
                if phys.sway > 0. {
                    orbit_rot -= phys.global_orbit_diff;
                }
            }

            bones[b].rot += orbit_rot;

            bones[b].scale *= parent_scale;
            bones[b].pos *= parent_scale;

            // orbit the parent
            bones[b].pos = rotate(&bones[b].pos, orbit_rot);

            bones[b].pos += parent_pos;
        }

        if let Some(ik_rot) = ik_rots.get(&(b as u32)) {
            bones[b].rot = *ik_rot;
        }

        // apply physics, if armature_bones is provided
        if let Some(phys) = physics.get(bones[b].physics_id as usize) {
            if phys.rot_damping > 0. {
                bones[b].rot = phys.global_rot;
            }
            if phys.pos_damping > 0. {
                bones[b].pos = phys.global_pos;
            }
            if phys.scale_damping > 0. {
                bones[b].scale = phys.global_scale;
            }
        }
    }
}

/// Always run this before `inheritance()`.`
pub fn reset_inheritance(constructed_bones: &mut Vec<Bone>, bones: &Vec<Bone>) {
    for b in 0..bones.len() {
        constructed_bones[b].pos = bones[b].pos;
        constructed_bones[b].rot = bones[b].rot;
        constructed_bones[b].scale = bones[b].scale;
    }
}

pub fn construct(armature: &mut Armature) {
    let const_bones = &mut armature.constructed_bones;

    // initialize constructed_bones
    if const_bones.len() == 0 {
        *const_bones = armature.bones.clone();
    } else {
        const_bones.sort_by(|a, b| a.id.partial_cmp(&b.id).unwrap());
    }

    // 1st inheritance pass
    reset_inheritance(const_bones, &armature.bones);
    inheritance(const_bones, HashMap::new(), &vec![]);

    // 2nd inheritance pass: inverse kinematics
    let mut ik_rots = HashMap::new();
    if !armature.baked_ik && armature.inverse_kinematics.len() > 0 {
        reset_inheritance(const_bones, &armature.bones);
        inheritance(const_bones, HashMap::new(), &vec![]);
        ik_rots = inverse_kinematics(const_bones, &armature.inverse_kinematics);
    }

    // 3rd inheritance pass: physics
    if armature.physics.len() > 0 {
        simulate_physics(const_bones, &mut armature.physics);
        reset_inheritance(const_bones, &armature.bones);
        inheritance(const_bones, ik_rots, &armature.physics);
    }

    // mesh deformation
    construct_verts(const_bones, &mut armature.visuals);
}

fn simulate_physics(constructed_bones: &mut Vec<Bone>, physics: &mut Vec<Physics>) {
    for b in 0..constructed_bones.len() {
        if constructed_bones[b].physics_id == -1 {
            continue;
        }
        let physics = &mut physics[constructed_bones[b].physics_id as usize];

        let s = Vec2::new(0.3, 0.3);
        let e = Vec2::new(0.6, 0.6);
        let const_bone = &constructed_bones[b];
        let prev_pos = physics.global_pos;

        // interpolate position
        if physics.pos_damping > 0. || physics.sway > 0. {
            let phys_pos = &mut physics.global_pos;
            let mut damping = Vec2::new(physics.pos_damping, physics.pos_damping);

            // ratio
            if physics.pos_ratio < 0. {
                damping.y *= 1. - physics.pos_ratio.abs();
            } else if physics.pos_ratio > 0. {
                damping.x *= 1. - physics.pos_ratio;
            }

            phys_pos.x = interpolate(2, damping.x as u32, phys_pos.x, const_bone.pos.x, s, e);
            phys_pos.y = interpolate(2, damping.y as u32, phys_pos.y, const_bone.pos.y, s, e);
        }

        // interpolate scale
        if physics.scale_damping > 0. {
            let phys_scale = &mut physics.global_scale;
            let mut damping = Vec2::new(physics.scale_damping, physics.scale_damping);

            // ratio
            if physics.scale_ratio < 0. {
                damping.y *= 1. - physics.scale_ratio.abs();
            } else if physics.pos_ratio > 0. {
                damping.x *= 1. - physics.scale_ratio;
            }

            phys_scale.x = interpolate(2, damping.x as u32, phys_scale.x, const_bone.scale.x, s, e);
            phys_scale.y = interpolate(2, damping.y as u32, phys_scale.y, const_bone.scale.y, s, e);
        }

        // interpolate rotation
        if physics.rot_damping > 0. {
            let rot = shortest_angle_delta(physics.global_rot, const_bone.rot);
            physics.global_rot += rot / physics.rot_damping;
        }

        // interpolate parent orbit (rot res, bounce, etc)
        let bones = &constructed_bones;
        let parent = bones.iter().find(|b| b.id == const_bone.parent_id as u32);
        if physics.sway > 0. && parent != None {
            // interpolate to the angle difference between bone and parent
            let diff = normalize(const_bone.pos - parent.unwrap().pos);
            let diff_angle = diff.y.atan2(diff.x);
            let mut orbit_buffer = shortest_angle_delta(physics.global_orbit, diff_angle);
            // apply bounce
            if physics.rot_bounce > 0. && physics.rot_bounce <= 1. {
                orbit_buffer += physics.global_orbit_vel / (2. - physics.rot_bounce);
                physics.global_orbit_vel = orbit_buffer;
            }
            physics.global_orbit += orbit_buffer / 10.;

            // swing orbit based on position momentum
            let vel = normalize(physics.global_pos - prev_pos);
            let angle = (-vel.y).atan2(-vel.x);
            let vel_rot = shortest_angle_delta(physics.global_orbit, angle);
            let strength = magnitude(physics.global_pos - prev_pos) / 1000.;
            physics.global_orbit += vel_rot * strength * physics.sway;

            physics.global_orbit_diff = diff_angle - physics.global_orbit;
        }
    }
}

pub fn construct_verts(bones: &mut Vec<Bone>, visuals: &mut Vec<Visuals>) {
    for b in 0..bones.len() {
        if bones[b].visuals_id == -1 {
            continue;
        }

        let visual = &mut visuals[bones[b].visuals_id as usize];

        // move vertex to main bone.
        // this will be overridden if vertex has a bind.
        for v in 0..visual.vertices.len() {
            visual.vertices[v].pos = visual.vertices[v].init_pos;
            visual.vertices[v].pos = inherit_vert(visual.vertices[v].pos, &bones[b]);
        }

        for bi in 0..visual.binds.len() {
            let b_id = visual.binds[bi].bone_id;
            if b_id == -1 {
                continue;
            }
            let bind_bone = bones
                .iter()
                .find(|bone| bone.id == b_id as u32)
                .unwrap()
                .clone();
            for v in 0..visual.binds[bi].verts.len() {
                let vert_id = visual.binds[bi].verts[v].id as usize;

                if !visual.binds[bi].is_path {
                    // weights
                    let weight = visual.binds[bi].verts[v].weight;
                    let end_pos = inherit_vert(visual.vertices[vert_id].init_pos, &bind_bone)
                        - visual.vertices[vert_id].pos;
                    visual.vertices[vert_id].pos += end_pos * weight;
                    continue;
                }

                // pathing:
                // Bone binds are treated as one continuous line.
                // Vertices will follow along this path.

                // get previous and next bone
                let binds = &visual.binds;
                let prev = if bi > 0 { bi - 1 } else { bi };
                let next = (bi + 1).min(binds.len() - 1);
                let prev_bone = bones
                    .iter()
                    .find(|bone| bone.id == binds[prev].bone_id as u32);
                let next_bone = bones
                    .iter()
                    .find(|bone| bone.id == binds[next].bone_id as u32);

                // get the average of normals between previous bone, this bone, and next bone
                let prev_dir = bind_bone.pos - prev_bone.unwrap().pos;
                let next_dir = next_bone.unwrap().pos - bind_bone.pos;
                let prev_normal = normalize(Vec2::new(-prev_dir.y, prev_dir.x));
                let next_normal = normalize(Vec2::new(-next_dir.y, next_dir.x));
                let average = prev_normal + next_normal;
                let normal_angle = average.y.atan2(average.x);

                // move vertex to bind bone, then just adjust it to 'bounce' off the line's surface
                visual.vertices[vert_id].pos = visual.vertices[vert_id].init_pos + bind_bone.pos;
                let rotated = rotate(
                    &(visual.vertices[vert_id].pos - bind_bone.pos),
                    normal_angle,
                );
                visual.vertices[vert_id].pos =
                    bind_bone.pos + (rotated * visual.binds[bi].verts[v].weight);
            }
        }
    }
}

pub fn inherit_vert(mut pos: Vec2, bone: &Bone) -> Vec2 {
    pos *= bone.scale;
    pos = rotate(&pos, bone.rot);
    pos += bone.pos;
    pos
}

fn magnitude(vec: Vec2) -> f32 {
    (vec.x * vec.x + vec.y * vec.y).sqrt()
}

fn normalize(vec: Vec2) -> Vec2 {
    let mag = magnitude(vec);
    if mag == 0. {
        return Vec2::default();
    }
    Vec2::new(vec.x / mag, vec.y / mag)
}

fn rotate(point: &Vec2, rot: f32) -> Vec2 {
    Vec2 {
        x: point.x * rot.cos() - point.y * rot.sin(),
        y: point.x * rot.sin() + point.y * rot.cos(),
    }
}

/// Interpolate an f32 value from the specified keyframe data.
pub fn interpolate_keyframes(
    field: &mut f32,
    prev_kf: &Keyframe,
    next_kf: &Keyframe,
    frame: u32,
    smooth_frames: u32,
) {
    let total_frames = next_kf.frame - prev_kf.frame;
    let current_frame = frame as u32 - prev_kf.frame as u32;

    let result = interpolate(
        current_frame as u32,
        total_frames,
        prev_kf.value,
        next_kf.value,
        next_kf.start_handle,
        next_kf.end_handle,
    );

    let z = Vec2::new(0., 0.);
    *field = interpolate(current_frame, smooth_frames as u32, *field, result, z, z);
}

fn interpolate(
    current: u32,
    max: u32,
    start_val: f32,
    end_val: f32,
    start_handle: Vec2,
    end_handle: Vec2,
) -> f32 {
    // snapping behavior for None transition preset
    if start_handle.y == 999. && end_handle.y == 999. {
        return start_val;
    }
    if max == 0 || current >= max {
        return end_val;
    }

    // solve for time (x axis) with Newton-Raphson
    let initial = current as f32 / max as f32;
    let mut t = initial;
    for _ in 0..5 {
        let x = cubic_bezier(t, start_handle.x, end_handle.x);
        let dx = cubic_bezier_derivative(t, start_handle.x, end_handle.x);
        if dx.abs() < 1e-5 {
            break;
        }
        t -= (x - initial) / dx;
        t = t.clamp(0., 1.);
    }

    let progress = cubic_bezier(t, start_handle.y, end_handle.y);
    start_val + (end_val - start_val) * progress
}

fn cubic_bezier(t: f32, p1: f32, p2: f32) -> f32 {
    let u = 1. - t;
    3. * u * u * t * p1 + 3. * u * t * t * p2 + t * t * t
}

fn cubic_bezier_derivative(t: f32, p1: f32, p2: f32) -> f32 {
    let u = 1. - t;
    return 3. * u * u * p1 + 6. * u * t * (p2 - p1) + 3. * t * t * (1. - p2);
}

/// Get rotations based on inverse kinematics.
/// Must be run between two `inheritance()`, with 2nd call using rotations from this.
pub fn inverse_kinematics(
    bones: &mut Vec<Bone>,
    inverse_kinematics: &Vec<InverseKinematics>,
) -> HashMap<u32, f32> {
    let mut ik_rot: HashMap<u32, f32> = HashMap::new();

    for family in inverse_kinematics {
        if family.target_id == -1 {
            continue;
        }
        let root = bones[family.bone_ids[0] as usize].pos;
        let target = bones[family.target_id as usize].pos;
        let mut family_bones: Vec<&mut Bone> = bones
            .iter_mut()
            .filter(|bone| family.bone_ids.contains(&bone.id))
            .collect();

        if family.mode == "FABRIK" {
            for _ in 0..10 {
                fabrik(&mut family_bones, root, target);
            }
        } else {
            arc_ik(&mut family_bones, root, target)
        }
        point_bones(bones, &family);
        apply_constraints(bones, &family);
        for b in 0..family.bone_ids.len() {
            if b == family.bone_ids.len() - 1 {
                continue;
            }
            ik_rot.insert(family.bone_ids[b], bones[family.bone_ids[b] as usize].rot);
        }
    }

    ik_rot
}

pub fn point_bones(bones: &mut Vec<Bone>, family: &InverseKinematics) {
    let end_bone = &bones[*family.bone_ids.last().unwrap() as usize];
    let mut tip_pos = end_bone.pos;
    for i in (0..family.bone_ids.len()).rev() {
        if i == family.bone_ids.len() - 1 {
            continue;
        }
        let bone = &mut bones[family.bone_ids[i] as usize];

        let dir = tip_pos - bone.pos;
        bone.rot = dir.y.atan2(dir.x);
        tip_pos = bone.pos;
    }
}

pub fn apply_constraints(bones: &mut Vec<Bone>, family: &InverseKinematics) {
    let root = bones[family.bone_ids[0] as usize].pos;
    let target = bones[family.target_id as usize].pos;
    let joint_dir = normalize(bones[family.bone_ids[1] as usize].pos - root);
    let base_dir = normalize(target - root);
    let dir = joint_dir.x * base_dir.y - base_dir.x * joint_dir.y;
    let base_angle = base_dir.y.atan2(base_dir.x);

    let cw = family.constraint == "Clockwise" && dir > 0.;
    let ccw = family.constraint == "CounterClockwise" && dir < 0.;
    if ccw || cw {
        for i in &family.bone_ids {
            bones[*i as usize].rot = -bones[*i as usize].rot + base_angle * 2.;
        }
    }
}

// https://www.youtube.com/watch?v=NfuO66wsuRg
pub fn fabrik(bones: &mut Vec<&mut Bone>, root: Vec2, target: Vec2) {
    // forward-reaching
    let mut next_pos: Vec2 = target;
    let mut next_length = 0.;
    for b in (0..bones.len()).rev() {
        let length = normalize(next_pos - bones[b].pos) * next_length;
        if b != 0 {
            next_length = magnitude(bones[b].pos - bones[b - 1].pos);
        }
        bones[b].pos = next_pos - length;
        next_pos = bones[b].pos;
    }

    // backward-reaching
    let mut prev_pos: Vec2 = root;
    let mut prev_length = 0.;
    for b in 0..bones.len() {
        let length = normalize(prev_pos - bones[b].pos) * prev_length;
        if b != bones.len() - 1 {
            prev_length = magnitude(bones[b].pos - bones[b + 1].pos);
        }
        bones[b].pos = prev_pos - length;
        prev_pos = bones[b].pos;
    }
}

pub fn arc_ik(bones: &mut Vec<&mut Bone>, root: Vec2, target: Vec2) {
    // determine where bones will be on the arc line (ranging from 0 to 1)
    let mut dist: Vec<f32> = vec![0.];

    let max_length = magnitude(bones.last().unwrap().pos - root);
    let mut curr_length = 0.;
    for b in 1..bones.len() {
        let length = magnitude(bones[b].pos - bones[b - 1].pos);
        curr_length += length;
        dist.push(curr_length / max_length);
    }

    let base = target - root;
    let base_angle = base.y.atan2(base.x);
    let base_mag = magnitude(base).min(max_length);
    let peak = max_length / base_mag;
    let valley = base_mag / max_length;

    for b in 1..bones.len() {
        bones[b].pos = Vec2::new(
            bones[b].pos.x * valley,
            root.y + (1. - peak) * (dist[b] * 3.14).sin() * base_mag,
        );

        let rotated = rotate(&(bones[b].pos - root), base_angle);
        bones[b].pos = rotated + root;
    }
}

pub fn format_frame(mut frame: u32, animation: &Animation, reverse: bool, is_loop: bool) -> u32 {
    let last_frame = animation.keyframes.last().unwrap().frame;

    if is_loop {
        frame %= last_frame + 1
    }

    if reverse {
        frame = last_frame - frame
    }

    frame
}

pub fn time_frame(time: Instant, animation: &Animation, reverse: bool, is_loop: bool) -> u32 {
    let elapsed = time.elapsed().as_millis() as f32 / 1e3 as f32;
    let frametime = 1. / animation.fps as f32;

    let mut frame = (elapsed / frametime) as u32;
    frame = format_frame(frame, animation, reverse, is_loop);

    frame
}

/// Flips the bone if either if its scale is negative.
pub fn check_flip(bone: &mut Bone, scale: Vec2) {
    let both = scale.x < 0. && scale.y < 0.;
    let either = scale.x < 0. || scale.y < 0.;
    if either && !both {
        bone.rot = -bone.rot;
    }
}

pub fn shortest_angle_delta(from: f32, to: f32) -> f32 {
    let pi = 3.141592653589793;
    let tau = pi * 2.0;
    let mut delta = to - from;
    while delta > pi {
        delta -= tau;
    }
    while delta < -pi {
        delta += tau;
    }
    delta
}
