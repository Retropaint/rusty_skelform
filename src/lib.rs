pub mod tests;

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

#[repr(C)]
#[derive(serde::Deserialize, Clone, Debug)]
pub struct Vertex {
    pub pos: Vec2,
    pub uv: Vec2,
    pub init_pos: Vec2,
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
#[rustfmt::skip]
pub struct Animation {
    #[serde(default)] 
    pub name: String,
    #[serde(default)] 
    pub fps: i32,
    #[serde(default)] 
    pub keyframes: Vec<Keyframe>,
}
#[derive(serde::Deserialize, Clone, Debug, Default)]
pub struct Bone {
    #[serde(default)]
    pub id: i32,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub parent_id: i32,
    #[serde(default)]
    pub tex: String,

    #[serde(default)]
    pub ik_family_id: i32,
    #[serde(default)]
    pub ik_constraint_str: JointConstraint,
    #[serde(default)]
    pub ik_constraint: i32,
    #[serde(default)]
    pub ik_mode_str: InverseKinematicsMode,
    #[serde(default)]
    pub ik_mode: i32,
    #[serde(default)]
    pub ik_target_id: i32,
    #[serde(default)]
    pub ik_bone_ids: Vec<i32>,

    #[serde(default)]
    pub vertices: Vec<Vertex>,
    #[serde(default)]
    pub indices: Vec<u32>,
    #[serde(default)]
    pub binds: Vec<BoneBind>,

    #[serde(default)]
    pub rot: f32,
    #[serde(default)]
    pub scale: Vec2,
    #[serde(default)]
    pub pos: Vec2,
    #[serde(default)]
    pub zindex: f32,

    #[serde(default)]
    pub init_rot: f32,
    #[serde(default)]
    pub init_scale: Vec2,
    #[serde(default)]
    pub init_pos: Vec2,
    #[serde(default)]
    pub init_ik_constraint: i32,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq, Default, Debug)]
pub struct BoneBind {
    #[serde(default)]
    pub bone_id: i32,
    #[serde(default)]
    pub is_path: bool,
    #[serde(default)]
    pub verts: Vec<BoneBindVert>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq, Default, Debug)]
pub struct BoneBindVert {
    #[serde(default)]
    pub id: i32,
    #[serde(default)]
    pub weight: f32,
}

#[derive(
    Eq, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize, Clone, Default, Debug,
)]
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

#[derive(PartialEq, serde::Serialize, serde::Deserialize, Clone, Default, Debug)]
pub struct Keyframe {
    #[serde(default)]
    pub frame: i32,
    #[serde(default)]
    pub bone_id: i32,
    #[serde(default)]
    pub element_str: AnimElement,
    #[serde(default)]
    pub element: i32,
    #[serde(default)]
    pub value: f32,
    #[serde(default)]
    pub value_str: String,
    #[serde(default)]
    pub transition: Transition,
    #[serde(skip)]
    pub label_top: f32,
}

#[derive(PartialEq, serde::Deserialize, Clone, Default, Debug)]
#[rustfmt::skip]
pub struct AnimBone {
    #[serde(default)] 
    pub id: i32,
    #[serde(default)] 
    pub fields: Vec<AnimField>,
}
#[derive(PartialEq, serde::Serialize, serde::Deserialize, Clone, Default, Debug)]
pub enum Transition {
    #[default]
    Linear,
    SineIn,
    SineOut,
}
#[derive(PartialEq, serde::Deserialize, Clone, Default, Debug)]
#[rustfmt::skip]
pub struct AnimField {
    #[serde(default)] 
    pub id: i32,
    #[serde(default)] 
    pub value: Vec2,
    #[serde(default)] 
    pub transition: Transition,
    #[serde(skip)]    
    pub label_top: f32,
}

#[derive(serde::Deserialize, Clone, Default, Debug)]
pub struct Style {
    #[serde(default)]
    pub id: i32,
    #[serde(default)]
    pub name: String,
    #[serde(skip)]
    pub active: bool,
    #[serde(default)]
    pub textures: Vec<Texture>,
}

#[derive(serde::Deserialize, Clone, Copy, Default, PartialEq, Debug)]
pub enum JointConstraint {
    #[default]
    None,
    Clockwise,
    CounterClockwise,
}

#[derive(serde::Serialize, serde::Deserialize, Copy, Clone, Default, PartialEq, Debug)]
pub enum InverseKinematicsMode {
    #[default]
    FABRIK,
    Arc,
}

#[derive(serde::Deserialize, Clone, Debug, Default)]
pub struct TexAtlas {
    #[serde(default)]
    pub filename: String,
    #[serde(default)]
    pub size: Vec2,
}

#[derive(serde::Deserialize, Clone, Debug, Default)]
pub struct Armature {
    #[serde(default)]
    pub ik_root_ids: Vec<i32>,
    #[serde(default)]
    pub bones: Vec<Bone>,
    #[serde(default)]
    pub animations: Vec<Animation>,
    #[serde(default)]
    pub textures: Vec<Texture>,
    #[serde(default)]
    pub styles: Vec<Style>,
    #[serde(default)]
    pub atlases: Vec<TexAtlas>,
    #[serde(skip)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Default, PartialEq)]
#[rustfmt::skip]
pub struct Metadata {
    pub last_anim: usize,
    pub last_frame: i32,
}

#[derive(serde::Deserialize, Clone, Default, Debug, PartialEq)]
pub struct Texture {
    #[serde(default)]
    pub offset: Vec2,
    #[serde(default)]
    pub size: Vec2,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub atlas_idx: i32,
}

/// Process bones with animations.
pub fn animate(
    bones: &mut Vec<Bone>,
    anims: &Vec<&Animation>,
    frames: &Vec<i32>,
    blend_frames: &Vec<i32>,
) {
    for a in 0..anims.len() {
        for b in 0..bones.len() {
            let keyframes = &anims[a].keyframes;
            let b_id = bones[b].id;
            interpolate_bone(&mut bones[b], keyframes, b_id, frames[a], blend_frames[a]);
        }
    }

    for bone in bones {
        reset_bone(bone, frames[0], blend_frames[0], anims);
    }
}

pub fn interpolate_bone(
    bone: &mut Bone,
    keyframes: &Vec<Keyframe>,
    bone_id: i32,
    frame: i32,
    blend_frame: i32,
) {
    interpolate_keyframes(0, &mut bone.pos.x, keyframes, bone_id, frame, blend_frame);
    interpolate_keyframes(1, &mut bone.pos.y, keyframes, bone_id, frame, blend_frame);
    interpolate_keyframes(2, &mut bone.rot, keyframes, bone_id, frame, blend_frame);
    interpolate_keyframes(3, &mut bone.scale.x, keyframes, bone_id, frame, blend_frame);
    interpolate_keyframes(4, &mut bone.scale.y, keyframes, bone_id, frame, blend_frame);
    let prev_frame = get_prev_frame(keyframes, frame, 7, bone.id);
    if prev_frame != usize::MAX {
        bone.ik_constraint = keyframes[prev_frame].value as i32;
    }

    let prev_frame = get_prev_frame(keyframes, frame, 6, bone.id);
    if prev_frame != usize::MAX {
        bone.tex = keyframes[prev_frame].value_str.clone();
    }
}

pub fn reset_bone(bone: &mut Bone, frame: i32, blend_frame: i32, anims: &Vec<&Animation>) {
    if !is_animated(bone.id, 0, anims) {
        bone.pos.x = interpolate(frame, blend_frame, bone.pos.x, bone.init_pos.x)
    }
    if !is_animated(bone.id, 1, anims) {
        bone.pos.y = interpolate(frame, blend_frame, bone.pos.y, bone.init_pos.y)
    }
    if !is_animated(bone.id, 2, anims) {
        bone.rot = interpolate(frame, blend_frame, bone.rot, bone.init_rot)
    }
    if !is_animated(bone.id, 3, anims) {
        bone.scale.x = interpolate(frame, blend_frame, bone.scale.x, bone.init_scale.x)
    }
    if !is_animated(bone.id, 4, anims) {
        bone.scale.y = interpolate(frame, blend_frame, bone.scale.y, bone.init_scale.y)
    }
    if !is_animated(bone.id, 7, anims) {
        bone.ik_constraint = bone.init_ik_constraint;
    }
}

pub fn setup_bone_textures(bones: &mut Vec<Bone>, styles: &Vec<&Style>) -> HashMap<i32, Texture> {
    let mut final_textures: HashMap<i32, Texture> = HashMap::new();
    for bone in bones {
        for style in styles {
            if let Some(tex) = style.textures.iter().find(|t| t.name == bone.tex) {
                final_textures.insert(bone.id, tex.clone());
                break;
            }
        }
    }

    final_textures
}

pub fn is_animated(bone_id: i32, el: i32, anims: &Vec<&Animation>) -> bool {
    for anim in anims {
        for kf in &anim.keyframes {
            if kf.bone_id == bone_id && kf.element == el {
                return true;
            }
        }
    }
    false
}

/// Apply child-parent inheritance.
/// Must be run twice, before and after `inverse_kinematics()`.
pub fn inheritance(bones: &mut Vec<Bone>, ik_rots: HashMap<i32, f32>) {
    for b in 0..bones.len() {
        if bones[b].parent_id != -1 {
            let parent = bones[bones[b].parent_id as usize].clone();

            bones[b].rot += parent.rot;
            bones[b].scale *= parent.scale;
            bones[b].pos *= parent.scale;
            bones[b].pos = rotate(&bones[b].pos, parent.rot);
            bones[b].pos += parent.pos;
        }

        if let Some(ik_rot) = ik_rots.get(&(b as i32)) {
            bones[b].rot = *ik_rot;
        }
    }
}

pub fn construct(armature: &Armature) -> Vec<Bone> {
    let mut inh_bones = armature.bones.clone();
    inheritance(&mut inh_bones, HashMap::new());
    let ik_rots = inverse_kinematics(&mut inh_bones, armature.ik_root_ids.clone());

    let mut final_bones = armature.bones.clone();
    inheritance(&mut final_bones, ik_rots);

    construct_verts(&mut final_bones);

    final_bones
}

pub fn construct_verts(bones: &mut Vec<Bone>) {
    for b in 0..bones.len() {
        let bone = bones[b].clone();

        // move vertex to main bone.
        // this will be overridden if vertex has a bind.
        for vert in &mut bones[b].vertices {
            vert.pos = inherit_vert(vert.pos, &bone);
        }

        for bi in 0..bones[b].binds.len() {
            let b_id = bones[b].binds[bi].bone_id;
            if b_id == -1 {
                continue;
            }
            let bind_bone = bones.iter().find(|bone| bone.id == b_id).unwrap().clone();
            let bind = bones[b].binds[bi].clone();
            for v in 0..bind.verts.len() {
                let vert_id = bind.verts[v].id as usize;

                if !bind.is_path {
                    // weights
                    let vert = &mut bones[b].vertices[vert_id];
                    let weight = bind.verts[v].weight;
                    let end_pos = inherit_vert(vert.init_pos, &bind_bone) - vert.pos;
                    vert.pos += end_pos * weight;
                    continue;
                }

                // pathing:
                // Bone binds are treated as one continuous line.
                // Vertices will follow along this path.

                // get previous and next bone
                let binds = &bones[b].binds;
                let prev = if bi > 0 { bi - 1 } else { bi };
                let next = (bi + 1).min(binds.len() - 1);
                let prev_bone = bones.iter().find(|bone| bone.id == binds[prev].bone_id);
                let next_bone = bones.iter().find(|bone| bone.id == binds[next].bone_id);

                // get the average of normals between previous bone, this bone, and next bone
                let prev_dir = bind_bone.pos - prev_bone.unwrap().pos;
                let next_dir = next_bone.unwrap().pos - bind_bone.pos;
                let prev_normal = normalize(Vec2::new(-prev_dir.y, prev_dir.x));
                let next_normal = normalize(Vec2::new(-next_dir.y, next_dir.x));
                let average = prev_normal + next_normal;
                let normal_angle = average.y.atan2(average.x);

                // move vertex to bind bone, then just adjust it to 'bounce' off the line's surface
                let vert = &mut bones[b].vertices[vert_id];
                vert.pos = vert.init_pos + bind_bone.pos;
                let rotated = rotate(&(vert.pos - bind_bone.pos), normal_angle);
                vert.pos = bind_bone.pos + (rotated * bind.verts[v].weight);
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

fn get_prev_frame(keyframes: &Vec<Keyframe>, frame: i32, element: i32, id: i32) -> usize {
    let mut prev = usize::MAX;
    for (i, kf) in keyframes.iter().enumerate() {
        if kf.frame <= frame && kf.element == element && kf.bone_id == id {
            prev = i;
        }
    }
    prev
}

fn get_next_frame(keyframes: &Vec<Keyframe>, frame: i32, element: i32, id: i32) -> usize {
    for (i, kf) in keyframes.iter().enumerate() {
        if kf.frame > frame && kf.element == element && kf.bone_id == id {
            return i;
        }
    }
    usize::MAX
}

/// Interpolate an f32 value from the specified keyframe data.
pub fn interpolate_keyframes(
    element: i32,
    field: &mut f32,
    keyframes: &Vec<Keyframe>,
    id: i32,
    frame: i32,
    blend_frames: i32,
) {
    let mut prev = get_prev_frame(keyframes, frame, element, id);
    let mut next = get_next_frame(keyframes, frame, element, id);

    // ensure both frames are pointing somewhere
    if prev == usize::MAX {
        prev = next;
    } else if next == usize::MAX {
        next = prev;
    }

    // if both are max, then the frame doesn't exist. Fallbackt to init value
    if prev == usize::MAX && next == usize::MAX {
        return;
    }

    let total_frames = keyframes[next].frame - keyframes[prev].frame;
    let current_frame = frame - keyframes[prev].frame;

    let result = interpolate(
        current_frame,
        total_frames,
        keyframes[prev].value,
        keyframes[next].value,
    );

    *field = interpolate(current_frame, blend_frames, *field, result);
}

fn interpolate(current: i32, max: i32, start_val: f32, end_val: f32) -> f32 {
    if max == 0 || current >= max {
        return end_val;
    }
    let interp = current as f32 / max as f32;
    let end = end_val - start_val;
    start_val + (end * interp)
}

/// Get rotations based on inverse kinematics.
/// Must be run between two `inheritance()`, with 2nd call using rotations from this.
pub fn inverse_kinematics(bones: &mut Vec<Bone>, ik_root_ids: Vec<i32>) -> HashMap<i32, f32> {
    let mut ik_rot: HashMap<i32, f32> = HashMap::new();

    for root_id in ik_root_ids {
        let family = bones[root_id as usize].clone();

        if family.ik_target_id == -1 {
            continue;
        }
        let root = bones[family.ik_bone_ids[0] as usize].pos;
        let target = bones[family.ik_target_id as usize].pos;
        let mut family_bones: Vec<&mut Bone> = bones
            .iter_mut()
            .filter(|bone| family.ik_bone_ids.contains(&bone.id))
            .collect();

        if family.ik_mode == 0 {
            for _ in 0..10 {
                fabrik(&mut family_bones, root, target);
            }
        } else {
            arc_ik(&mut family_bones, root, target)
        }

        point_bones(bones, family.clone());

        apply_constraints(bones, family.clone());

        for b in 0..family.ik_bone_ids.len() {
            if b == family.ik_bone_ids.len() - 1 {
                continue;
            }
            ik_rot.insert(
                family.ik_bone_ids[b],
                bones[family.ik_bone_ids[b] as usize].rot,
            );
        }
    }

    ik_rot
}

pub fn point_bones(bones: &mut Vec<Bone>, family: Bone) {
    let end_bone = &bones[*family.ik_bone_ids.last().unwrap() as usize];
    let mut tip_pos = end_bone.pos;
    for i in (0..family.ik_bone_ids.len()).rev() {
        if i == family.ik_bone_ids.len() - 1 {
            continue;
        }
        let bone = &mut bones[family.ik_bone_ids[i] as usize];

        let dir = tip_pos - bone.pos;
        bone.rot = dir.y.atan2(dir.x);
        tip_pos = bone.pos;
    }
}

pub fn apply_constraints(bones: &mut Vec<Bone>, family: Bone) {
    let root = bones[family.ik_bone_ids[0] as usize].pos;
    let target = bones[family.ik_target_id as usize].pos;
    let joint_dir = normalize(bones[family.ik_bone_ids[1] as usize].pos - root);
    let base_dir = normalize(target - root);
    let dir = joint_dir.x * base_dir.y - base_dir.x * joint_dir.y;
    let base_angle = base_dir.y.atan2(base_dir.x);

    let cw = family.ik_constraint == 1 && dir > 0.;
    let ccw = family.ik_constraint == 2 && dir < 0.;
    if ccw || cw {
        for i in &family.ik_bone_ids {
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
        let mut length = normalize(next_pos - bones[b].pos) * next_length;
        if length.x.is_nan() {
            length = Vec2::new(0., 0.);
        }
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
        let mut length = normalize(prev_pos - bones[b].pos) * prev_length;
        if length.x.is_nan() {
            length = Vec2::new(0., 0.);
        }
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

pub fn format_frame(mut frame: i32, animation: &Animation, reverse: bool, is_loop: bool) -> i32 {
    let last_frame = animation.keyframes.last().unwrap().frame;

    if is_loop {
        frame %= last_frame + 1
    }

    if reverse {
        frame = last_frame - frame
    }

    frame
}

pub fn time_frame(time: Instant, animation: &Animation, reverse: bool, is_loop: bool) -> i32 {
    let elapsed = time.elapsed().as_millis() as f32 / 1e3 as f32;
    let frametime = 1. / animation.fps as f32;

    let mut frame = (elapsed / frametime) as i32;
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
