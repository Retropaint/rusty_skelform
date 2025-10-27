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
impl std::fmt::Display for Vec2 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let decimal_places = 3;

        let mut p = 0;
        let mut dp = 1.;
        while p < decimal_places {
            dp *= 10.;
            p += 1;
        }

        write!(
            f,
            "{}, {}",
            (self.x * dp).trunc() / dp,
            (self.y * dp).trunc() / dp
        )
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
#[rustfmt::skip]
pub struct Bone {
    #[serde(default)] 
    pub id: i32,
    #[serde(default, rename="_name")] 
    pub name: String,

    #[serde(default = "default_neg_one")] 
    pub parent_id: i32,
    #[serde(default)] 
    pub style_ids: Vec<i32>,
    #[serde(default)] 
    pub tex_idx: i32,

    #[serde(default)] 
    pub vertices: Vec<Vertex>,
    #[serde(default)] 
    pub indices: Vec<u32>,

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
    VertPositionX,
    VertPositionY,
    Texture,
}

#[derive(PartialEq, serde::Serialize, serde::Deserialize, Clone, Default, Debug)]
pub struct Keyframe {
    #[serde(default)]
    pub frame: i32,
    #[serde(default)]
    pub bone_id: i32,
    #[serde(default, rename = "_element")]
    pub element: AnimElement,
    #[serde(default)]
    pub element_id: i32,
    #[serde(default = "default_neg_one")]
    pub vert_id: i32,
    #[serde(default)]
    pub value: f32,
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
    #[serde(default, rename = "_name")]
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

#[derive(serde::Deserialize, Clone, Debug)]
pub struct IkFamily {
    pub constraint: JointConstraint,
    pub target_id: i32,
    pub bone_ids: Vec<i32>,
}

#[derive(serde::Deserialize, Clone, Debug, Default)]
pub struct Armature {
    #[serde(default)]
    pub texture_size: Vec2,
    #[serde(default)]
    pub bones: Vec<Bone>,
    #[serde(default)]
    pub animations: Vec<Animation>,
    #[serde(default)]
    pub textures: Vec<Texture>,
    #[serde(default)]
    pub styles: Vec<Style>,
    #[serde(default)]
    pub ik_families: Vec<IkFamily>,
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
    #[serde(default, rename = "_name")]
    pub name: String,
    #[serde(skip)]
    pub pixels: Vec<u8>,
}

/// Process bones with animations.
pub fn animate(bones: &mut Vec<Bone>, anim: &Animation, frame: i32, blend_frames: i32) {
    for bone in bones {
        let keyframes = &anim.keyframes;

        #[rustfmt::skip]
        macro_rules! animate {
            ($element:expr, $field:expr) => {
                interpolate_keyframes($element, &mut $field, keyframes, bone.id, frame, blend_frames)
            };
        }

        animate!(AnimElement::PositionX, bone.pos.x);
        animate!(AnimElement::PositionY, bone.pos.y);
        animate!(AnimElement::Rotation, bone.rot);
        animate!(AnimElement::ScaleX, bone.scale.x);
        animate!(AnimElement::ScaleY, bone.scale.y);
    }
}

/// Reset bones back to default states, if they haven't been animated.
/// Must be called after `animate()` with the same animations provided.
/// `frame` must be first anim frame.
pub fn reset_bones(bones: &mut Vec<Bone>, anims: &Vec<&Animation>, frame: i32, blend_frames: i32) {
    for bone in bones {
        macro_rules! attempt {
            ($eid:expr, $field:expr, $init:expr) => {
                if !has_kf(bone.id, $eid, anims) {
                    $field = interpolate(frame, blend_frames, $field, $init)
                }
            };
        }

        attempt!(AnimElement::PositionX, bone.pos.x, bone.init_pos.x);
        attempt!(AnimElement::PositionY, bone.pos.y, bone.init_pos.y);
        attempt!(AnimElement::Rotation, bone.rot, bone.init_rot);
        attempt!(AnimElement::ScaleX, bone.scale.x, bone.init_scale.x);
        attempt!(AnimElement::ScaleY, bone.scale.y, bone.init_scale.y);
    }
}

pub fn has_kf(bone_id: i32, el: AnimElement, anims: &Vec<&Animation>) -> bool {
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

fn magnitude(vec: Vec2) -> f32 {
    (vec.x * vec.x + vec.y * vec.y).sqrt()
}

fn normalize(vec: Vec2) -> Vec2 {
    let mag = magnitude(vec);
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
    element: AnimElement,
    field: &mut f32,
    keyframes: &Vec<Keyframe>,
    id: i32,
    frame: i32,
    blend_frames: i32,
) {
    let mut prev = usize::MAX;
    let mut next = usize::MAX;

    // get start frame
    for (i, kf) in keyframes.iter().enumerate() {
        if kf.frame < frame && kf.element == element && kf.bone_id == id {
            prev = i;
        }
    }

    // get end frame
    for (i, kf) in keyframes.iter().enumerate() {
        if kf.frame >= frame && kf.element == element && kf.bone_id == id {
            next = i;
            break;
        }
    }

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

pub fn find_bone(id: i32, bones: &Vec<Bone>) -> Option<&Bone> {
    for bone in bones {
        if bone.id == id {
            return Some(bone);
        }
    }
    None
}

/// Get rotations based on inverse kinematics.
/// Must be run between two `inheritance()`, with 2nd call using rotations from this.
pub fn inverse_kinematics(bones: &mut Vec<Bone>, ik_families: &Vec<IkFamily>) -> HashMap<i32, f32> {
    let mut ik_rot: HashMap<i32, f32> = HashMap::new();

    for family in ik_families {
        if family.target_id == -1 {
            continue;
        }
        let root = bones[family.bone_ids[0] as usize].pos;
        let base_line = normalize(bones[family.target_id as usize].pos - root);
        let base_angle = base_line.y.atan2(base_line.x);

        // forward reaching
        let mut next_pos = bones[family.target_id as usize].pos;
        let mut next_length = 0.;
        for i in (0..family.bone_ids.len()).rev() {
            macro_rules! bone {
                () => {
                    bones[family.bone_ids[i] as usize]
                };
            }

            let mut length = Vec2::new(0., 0.);
            if i != family.bone_ids.len() - 1 {
                length = normalize(next_pos - bone!().pos) * next_length;
            }

            if i != 0 {
                let next_bone = &bones[family.bone_ids[i - 1] as usize];
                next_length = magnitude(bone!().pos - next_bone.pos);
            }
            bone!().pos = next_pos - length;

            next_pos = bone!().pos;
        }

        // backward reaching
        let mut prev_pos = root;
        let mut prev_length = 0.;
        for i in 0..family.bone_ids.len() {
            if family.target_id == -1 {
                continue;
            }
            macro_rules! bone {
                () => {
                    bones[family.bone_ids[i] as usize]
                };
            }

            let mut length = Vec2::new(0., 0.);
            if i != 0 {
                length = normalize(prev_pos - bone!().pos) * prev_length;
            }

            if i != family.bone_ids.len() - 1 {
                let prev_bone = &bones[family.bone_ids[i + 1] as usize];
                prev_length = magnitude(bone!().pos - prev_bone.pos);
            }

            bone!().pos = prev_pos - length;

            if i != 0
                && i != family.bone_ids.len() - 1
                && family.constraint != JointConstraint::None
            {
                let joint_line = normalize(prev_pos - bone!().pos);
                let joint_angle = joint_line.y.atan2(joint_line.x) - base_angle;

                let constraint_min;
                let constraint_max;
                if family.constraint == JointConstraint::Clockwise {
                    constraint_min = -3.14;
                    constraint_max = 0.;
                } else {
                    constraint_min = 0.;
                    constraint_max = 3.14;
                }

                if joint_angle > constraint_max || joint_angle < constraint_min {
                    let push_angle = -joint_angle * 2.;
                    let new_point = rotate(&(bone!().pos - prev_pos), push_angle);
                    bone!().pos = new_point + prev_pos;
                }
            }

            prev_pos = bone!().pos;
        }

        let end_bone = &bones[*family.bone_ids.last().unwrap() as usize];
        let mut tip_pos = end_bone.pos;
        for i in (0..family.bone_ids.len()).rev() {
            if i == family.bone_ids.len() - 1 {
                continue;
            }
            macro_rules! bone {
                () => {
                    bones[family.bone_ids[i] as usize]
                };
            }

            let dir = tip_pos - bone!().pos;
            bone!().rot = dir.y.atan2(dir.x);
            tip_pos = bone!().pos;
        }
    }

    for family in ik_families {
        for i in 0..family.bone_ids.len() {
            if i == family.bone_ids.len() - 1 {
                continue;
            }
            ik_rot.insert(family.bone_ids[i], bones[family.bone_ids[i] as usize].rot);
        }
    }

    ik_rot
}

pub fn format_frame(
    mut frame: i32,
    animation: &Animation,
    reverse: bool,
    should_loop: bool,
) -> i32 {
    let last_frame = animation.keyframes.last().unwrap().frame;

    if should_loop {
        frame %= last_frame + 1
    }

    if reverse {
        frame = last_frame - frame
    }

    frame
}

pub fn time_frame(time: Instant, animation: &Animation, reverse: bool, should_loop: bool) -> i32 {
    let elapsed = time.elapsed().as_millis() as f32 / 1e3 as f32;
    let frametime = 1. / animation.fps as f32;

    let mut frame = (elapsed / frametime) as i32;
    frame = format_frame(frame, animation, reverse, should_loop);

    frame
}

fn default_neg_one() -> i32 {
    -1
}
