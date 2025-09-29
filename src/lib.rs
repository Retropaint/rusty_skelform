use std::{collections::HashMap, time::Instant};

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
    pub parent_idx: i32,
    #[serde(default)] 
    pub style_idxs: Vec<i32>,
    #[serde(default = "default_neg_one")] 
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
    pub pivot: Vec2,
    #[serde(default)] 
    pub zindex: f32,

    /// used to properly offset bone's movement to counteract it's parent
    #[serde(skip)] pub parent_rot: f32,
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
    PivotX,
    PivotY,
    Zindex,
    VertPositionX,
    VertPositionY,
    Texture,
}

#[derive(PartialEq, serde::Serialize, serde::Deserialize, Clone, Default, Debug)]
#[rustfmt::skip]
pub struct Keyframe {
    #[serde(default)] 
    pub frame: i32,
    #[serde(default)] 
    pub bone_id: i32,
    #[serde(default)] 
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
    #[serde(skip)]
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
    pub target_idx: i32,
    pub bone_idxs: Vec<i32>,
}

#[derive(serde::Deserialize, Clone, Debug, Default)]
pub struct Armature {
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

#[derive(serde::Deserialize, Clone, Debug)]
pub struct SkelformRoot {
    pub texture_size: Vec2,
    pub armature: Armature,
}

#[derive(serde::Deserialize, Clone, Default, Debug)]
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

/// Returned from animate(), providing all animation data of a single bone (and then some) in a single frame.
#[derive(Clone)]
pub struct Prop {
    pub name: String,
    pub parent_id: i32,

    pub tex_idx: i32,

    /// Bone transforms
    pub pos: Vec2,
    pub scale: Vec2,
    pub rot: f32,
    pub pivot: Vec2,

    /// Mesh data
    pub is_mesh: bool,
    pub verts: Vec<Vertex>,

    /// z-index. Lower values should render behind higher.
    pub zindex: i32,
}

/// Modify animation based on time, and return the appropriate frame.
pub fn get_frame_by_time(anim: &mut Animation, time: Instant, speed: f32) -> i32 {
    // modify frames to simulate speeds
    for kf in &mut anim.keyframes {
        kf.frame = (kf.frame as f32 * (1. / speed.abs())) as i32;
    }

    let elapsed = time.elapsed().as_millis() as f32 / 1e3 as f32;
    let frametime = 1. / anim.fps as f32;
    let mut frame = (elapsed / frametime) as i32;

    // reverse animation if speed is negative
    if speed < 0. {
        frame = anim.keyframes.last().unwrap().frame - frame;
    }

    frame
}

/// Process an animation at the specified frame.
///
/// `after_animate` is a closure that runs immediately after animations, and before inheritence of a bone's parent properties.
///
/// last_anim idx and frame are used for blending.
pub fn animate(
    armature: &mut Armature,
    anim_idx: usize,
    mut frame: i32,
    should_loop: bool,
) -> Vec<Bone> {
    let anim = armature.animations[anim_idx].clone();
    let last_frame = anim.keyframes.last().unwrap().frame;

    if should_loop && last_frame != 0 {
        frame %= last_frame;
    }

    armature.metadata.last_frame = frame;

    let mut props: Vec<Bone> = Vec::new();

    for bone in &armature.bones {
        props.push(bone.clone());

        macro_rules! prop {
            () => {
                props.last_mut().unwrap()
            };
        }

        // animate prop
        macro_rules! animate {
            ($element:expr, $vert_id:expr, $og_value:expr) => {
                animate_f32(
                    &anim.keyframes,
                    prop!().id,
                    frame,
                    $element,
                    $vert_id,
                    $og_value,
                )
            };
        }

        #[rustfmt::skip] {
                prop!().pos.x   += animate!(AnimElement::PositionX, -1, 0.).0;
                prop!().pos.y   += animate!(AnimElement::PositionY, -1, 0.).0;
                prop!().rot     += animate!(AnimElement::Rotation,  -1, 0.).0;
                prop!().scale.x *= animate!(AnimElement::ScaleX,    -1, 1.).0;
                prop!().scale.y *= animate!(AnimElement::ScaleY,    -1, 1.).0;
                prop!().pivot.x += animate!(AnimElement::PivotX,    -1, 0.).0;
                prop!().pivot.y += animate!(AnimElement::PivotY,    -1, 0.).0;
            };

        for v in 0..prop!().vertices.len() {
            prop!().vertices[v].pos.x += animate!(AnimElement::VertPositionX, v as i32, 0.).0;
            prop!().vertices[v].pos.y += animate!(AnimElement::VertPositionY, v as i32, 0.).0;
        }

        let tex_frame = animate!(AnimElement::Texture, -1, 0.).1;
        if tex_frame != usize::MAX {
            let prev_tex_idx = anim.keyframes[tex_frame].value;
            prop!().tex_idx = prev_tex_idx as i32;
        }
    }
    props
}

pub fn inheritance(bones: &mut Vec<Bone>, ik_rots: HashMap<i32, f32>) {
    for b in 0..bones.len() {
        if bones[b].parent_idx != -1 {
            let parent = bones[bones[b].parent_idx as usize].clone();
            let parent_rot = parent.rot;

            bones[b].rot += parent.rot;
            bones[b].scale *= parent.scale;
            bones[b].pos *= parent.scale;
            bones[b].pos = Vec2::new(
                bones[b].pos.x * parent_rot.cos() - bones[b].pos.y * parent_rot.sin(),
                bones[b].pos.x * parent_rot.sin() + bones[b].pos.y * parent_rot.cos(),
            );
            bones[b].pos += parent.pos;
        }

        if let Some(ik_rot) = ik_rots.get(&(b as i32)) {
            bones[b].rot = *ik_rot;
        }
    }
}

pub fn magnitude(vec: Vec2) -> f32 {
    (vec.x * vec.x + vec.y * vec.y).sqrt()
}

pub fn normalize(vec: Vec2) -> Vec2 {
    let mag = magnitude(vec);
    Vec2::new(vec.x / mag, vec.y / mag)
}

pub fn rotate(point: &Vec2, rot: f32) -> Vec2 {
    Vec2 {
        x: point.x * rot.cos() - point.y * rot.sin(),
        y: point.x * rot.sin() + point.y * rot.cos(),
    }
}

pub fn inverse_kinematics(bones: &mut Vec<Bone>, ik_families: &Vec<IkFamily>) -> HashMap<i32, f32> {
    let mut ik_rot: HashMap<i32, f32> = HashMap::new();

    for family in ik_families {
        let base_line = normalize(
            bones[family.target_idx as usize].pos - bones[family.bone_idxs[0] as usize].pos,
        );
        let base_angle = base_line.y.atan2(base_line.x);

        // forward reaching
        let mut next_pos = bones[family.target_idx as usize].pos;
        let mut next_length = 0.;
        for i in (0..family.bone_idxs.len()).rev() {
            macro_rules! bone {
                () => {
                    bones[family.bone_idxs[i] as usize]
                };
            }

            let mut length = normalize(next_pos - bone!().pos) * next_length;
            if length.x.is_nan() {
                length = Vec2::new(0., 0.);
            }

            if i != 0 {
                let next_bone = &bones[family.bone_idxs[i - 1] as usize];
                next_length = magnitude(bone!().pos - next_bone.pos);
            }
            bone!().pos = next_pos - length;

            if i != 0
                && i != family.bone_idxs.len() - 1
                && family.constraint != JointConstraint::None
            {
                let joint_line = normalize(next_pos - bone!().pos);
                let joint_angle = joint_line.y.atan2(joint_line.x) - base_angle;

                let constraint_min;
                let constraint_max;
                if family.constraint == JointConstraint::Clockwise {
                    constraint_min = 0.;
                    constraint_max = 3.14;
                } else {
                    constraint_min = -3.14;
                    constraint_max = 0.;
                }

                if joint_angle > constraint_max || joint_angle < constraint_min {
                    let push_angle = -joint_angle * 2.;
                    let new_point = rotate(&(bone!().pos - next_pos), push_angle);
                    bone!().pos = new_point + next_pos;
                }
            }

            next_pos = bone!().pos;
        }

        // backward reaching
        let mut prev_pos = bones[family.bone_idxs[0] as usize].pos;
        let mut prev_length = 0.;
        for i in 0..family.bone_idxs.len() {
            macro_rules! bone {
                () => {
                    bones[family.bone_idxs[i] as usize]
                };
            }
            let mut length = normalize(prev_pos - bone!().pos) * prev_length;
            if length.x.is_nan() {
                length = Vec2::new(0., 0.);
            }

            if i != family.bone_idxs.len() - 1 {
                let prev_bone = &bones[family.bone_idxs[i + 1] as usize];
                prev_length = magnitude(bone!().pos - prev_bone.pos);
            }

            bone!().pos = prev_pos - length;
            prev_pos = bone!().pos;
        }

        let end_bone = &bones[*family.bone_idxs.last().unwrap() as usize];
        let mut tip_pos = end_bone.pos;
        for i in (0..family.bone_idxs.len()).rev() {
            macro_rules! bone {
                () => {
                    bones[family.bone_idxs[i] as usize]
                };
            }
            if i == family.bone_idxs.len() - 1 {
                continue;
            }
            let dir = tip_pos - bone!().pos;
            bone!().rot = dir.y.atan2(dir.x);
            tip_pos = bone!().pos;

            ik_rot.insert(family.bone_idxs[i], bone!().rot);

            println!("{}", bone!().rot);
        }
    }

    ik_rot
}

/// Interpolate an f32 value from the specified keyframe data.
pub fn animate_f32(
    keyframes: &Vec<Keyframe>,
    id: i32,
    frame: i32,
    element: AnimElement,
    vert_id: i32,
    og_value: f32,
) -> (f32, usize, usize) {
    let mut prev = usize::MAX;
    let mut next = usize::MAX;

    // get start frame
    for (i, kf) in keyframes.iter().enumerate() {
        if kf.frame > frame {
            break;
        } else if kf.element == element && kf.bone_id == id && kf.vert_id == vert_id {
            prev = i;
        }
    }

    // get end frame
    for (i, kf) in keyframes.iter().enumerate() {
        if kf.frame >= frame && kf.element == element && kf.bone_id == id && kf.vert_id == vert_id {
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

    // if both are max, then the frame doesn't exist. Keep original value
    if prev == usize::MAX && next == usize::MAX {
        return (og_value, usize::MAX, usize::MAX);
    }

    let mut total_frames = keyframes[next].frame - keyframes[prev].frame;
    // Tweener doesn't accept duration of 0
    if total_frames == 0 {
        total_frames = 1;
    }

    let current_frame = frame - keyframes[prev].frame;

    // run the transition
    macro_rules! transition {
        ($tweener:expr) => {
            $tweener(keyframes[prev].value, keyframes[next].value, total_frames)
                .move_to(current_frame)
        };
    }

    let current = match keyframes[next].transition {
        Transition::Linear => transition!(tween::Tweener::linear),
        Transition::SineIn => transition!(tween::Tweener::sine_in),
        Transition::SineOut => transition!(tween::Tweener::sine_out),
    };

    (current, prev, next)
}

pub fn find_bone(id: i32, bones: &Vec<Bone>) -> Option<&Bone> {
    for bone in bones {
        if bone.id == id {
            return Some(bone);
        }
    }
    None
}

fn default_neg_one() -> i32 {
    -1
}
