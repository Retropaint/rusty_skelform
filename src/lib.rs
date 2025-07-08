use std::{
    f32::consts::PI,
    ops::{AddAssign, MulAssign},
    time::Instant,
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

impl MulAssign for Vec2 {
    fn mul_assign(&mut self, other: Vec2) {
        self.x *= other.x;
        self.y *= other.y;
    }
}

impl MulAssign<f32> for Vec2 {
    fn mul_assign(&mut self, other: f32) {
        self.x *= other;
        self.y *= other;
    }
}

impl std::ops::DivAssign for Vec2 {
    fn div_assign(&mut self, other: Vec2) {
        self.x /= other.x;
        self.y /= other.y;
    }
}

impl std::ops::DivAssign<f32> for Vec2 {
    fn div_assign(&mut self, other: f32) {
        self.x /= other;
        self.y /= other;
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, other: Vec2) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl std::ops::SubAssign for Vec2 {
    fn sub_assign(&mut self, other: Vec2) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::Div for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
        }
    }
}

impl std::ops::Div<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl std::ops::Mul for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Vec2) -> Self {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::Sub<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        Self {
            x: self.x - rhs,
            y: self.y - rhs,
        }
    }
}

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
    #[serde(default)] pub name: String,
    #[serde(default)] pub fps: i32,
    #[serde(default)] pub keyframes: Vec<Keyframe>,
}
#[derive(serde::Deserialize, Clone, Debug, Default)]
#[rustfmt::skip]
pub struct Bone {
    #[serde(default)] pub id: i32,
    #[serde(default)] pub name: String,

    #[serde(default = "default_neg_one")] 
    pub parent_id: i32,
    #[serde(default = "default_neg_one")] 
    pub tex_idx: i32,

    #[serde(default)] pub vertices: Vec<Vertex>,
    #[serde(default)] pub indices: Vec<u32>,

    #[serde(default)] pub rot: f32,
    #[serde(default)] pub scale: Vec2,
    #[serde(default)] pub pos: Vec2,
    #[serde(default)] pub pivot: Vec2,
    #[serde(default)] pub zindex: f32,

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
    #[serde(default)] pub frame: i32,
    #[serde(default)] pub bone_id: i32,
    #[serde(default)] pub element: AnimElement,
    #[serde(default)] pub element_id: i32,
    #[serde(default = "default_neg_one")] pub vert_id: i32,
    #[serde(default)] pub value: f32,
    #[serde(default)] pub transition: Transition,
    #[serde(skip)]    pub label_top: f32,
}

#[derive(PartialEq, serde::Deserialize, Clone, Default, Debug)]
#[rustfmt::skip]
pub struct AnimBone {
    #[serde(default)] pub id: i32,
    #[serde(default)] pub fields: Vec<AnimField>,
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
    #[serde(default)] pub id: i32,

    // If the next field is related to this, connect is true.
    //
    // Example: Color is a vec4 value (RGBA), so the first field
    // is for RG, while second is for BA. The first field's
    // connect is true, while the second one's is false as it does not connect
    // to the field after it.
    //
    // This can be chained to have as many even-numbered vecs as possible.
    #[serde(default)] pub connect: bool,

    #[serde(default)] pub value: Vec2,

    #[serde(default)] pub transition: Transition,

    #[serde(skip)]    pub label_top: f32,
}

#[derive(serde::Deserialize, Clone, Debug, Default)]
pub struct Armature {
    #[serde(default)]
    pub bones: Vec<Bone>,
    #[serde(default)]
    pub animations: Vec<Animation>,
    #[serde(default)]
    pub textures: Vec<Texture>,

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
    pub armatures: Vec<Armature>,
}

#[derive(serde::Deserialize, Clone, Default, Debug)]
pub struct Texture {
    #[serde(default)]
    pub offset: Vec2,
    #[serde(default)]
    pub size: Vec2,
    #[serde(default)]
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

/// Animate using time to identify frame.
/// This is the recommended way to animate.
/// `should_loop` - Loop animation.
pub fn animate_by_time<T: FnOnce(&Bone, &mut Bone) + Copy>(
    armature: &mut Armature,
    anim_idx: usize,
    time: Instant,
    should_loop: bool,
    speed: f32,
    after_animate: T,
    mut last_anim_idx: usize,
    last_anim_frame: i32,
) -> (Vec<Bone>, i32) {
    if last_anim_idx == usize::MAX {
        last_anim_idx = anim_idx;
    }
    // clone the animation, or initialize empty one
    let mut anim = Animation::default();
    if anim_idx < armature.animations.len() {
        anim = armature.animations[anim_idx].clone()
    } else {
        anim.keyframes.push(Keyframe::default())
    }

    // modify frames to simulate speeds
    for kf in &mut anim.keyframes {
        kf.frame = (kf.frame as f32 * (1. / speed.abs())) as i32;
    }

    let elapsed = time.elapsed().as_millis() as f32 / 1e3 as f32;
    let frametime = 1. / anim.fps as f32;
    let mut frame = (elapsed / frametime) as i32;
    let last_frame = anim.keyframes.last().unwrap().frame;
    if should_loop && last_frame != 0 {
        frame %= last_frame;
    }

    // reverse animation if speed is negative
    if speed < 0. {
        frame = last_frame - frame;
    }

    (
        animate(
            armature,
            anim_idx,
            &mut anim,
            frame,
            should_loop,
            after_animate,
            last_anim_idx,
            last_anim_frame,
        ),
        frame,
    )
}

/// Process an animation at the specified frame.
///
/// `after_animate` is a closure that runs immediately after animations, and before inheritence of a bone's parent properties.
///
/// last_anim idx and frame are used for blending.
pub fn animate<T: FnOnce(&Bone, &mut Bone) + Copy>(
    armature: &mut Armature,
    anim_idx: usize,
    raw_anim: &Animation,
    mut frame: i32,
    should_loop: bool,
    after_animate: T,
    last_anim_idx: usize,
    last_anim_frame: i32,
) -> Vec<Bone> {
    let mut anim = raw_anim.clone();
    let last_frame = anim.keyframes.last().unwrap().frame;

    if frame < 0 {
        frame = 0;
    }
    if frame > last_frame - 1 {
        frame = last_frame - 1;
    }

    if should_loop && last_frame != 0 {
        frame %= last_frame;
    }

    // record last animation, after this one loops
    if frame == 0 && armature.metadata.last_frame == last_frame - 1 {
        armature.metadata.last_anim = anim_idx;
    }

    // simulate blending, by injecting last animation data into this one
    if armature.metadata.last_anim != anim_idx {
        // remove 0th frame keyframe that aren't part of the last animation
        let mut removed = false;
        let mut curr = 0;
        while removed {
            if anim.keyframes[curr].frame != 0 {
                removed = true;
            }
            let mut exists = false;
            for kf in &armature.animations[last_anim_idx].keyframes {
                if kf.bone_id == anim.keyframes[curr].bone_id {
                    exists = true;
                    curr += 1;
                }
            }
            if !exists {
                anim.keyframes.remove(curr);
            }
        }

        // inject last animation's keyframes to this one, but at 0th frame
        for mut kf in armature.animations[last_anim_idx].keyframes.clone() {
            if kf.frame != last_anim_frame {
                continue;
            }
            kf.frame = 0;
            anim.keyframes.insert(0, kf.clone());
        }

        // get very next keyframe
        let mut next_frame = 0;
        while anim.keyframes[next_frame].frame == 0 {
            next_frame += 1;
        }
        next_frame = anim.keyframes[next_frame].frame as usize;

        // add default keyframes for elements that don't exist, so they can be animated
        for kf in &armature.animations[last_anim_idx].keyframes {
            let mut already = false;
            for kf2 in &anim.keyframes {
                if kf2.frame == next_frame as i32 && kf2.element == kf.element {
                    already = true;
                    break;
                }
            }
            if already {
                continue;
            }

            let mut new_kf = kf.clone();
            new_kf.frame = next_frame as i32;
            new_kf.value = 0.;
            anim.keyframes.push(new_kf);
            anim.keyframes.sort_by(|a, b| a.frame.cmp(&b.frame));
        }
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

        let og_prop = prop!().clone();

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

        after_animate(&og_prop, prop!());

        if prop!().parent_id == -1 {
            continue;
        }

        // inherit transform from parent

        let parent = find_bone(prop!().parent_id, &props).unwrap().clone();
        let parent_rot = parent.rot;

        prop!().rot += parent.rot;
        prop!().scale *= parent.scale;
        prop!().pos *= parent.scale;
        prop!().pos = Vec2::new(
            prop!().pos.x * parent_rot.cos() - prop!().pos.y * parent_rot.sin(),
            prop!().pos.x * parent_rot.sin() + prop!().pos.y * parent_rot.cos(),
        );
        prop!().pos += parent.pos;
    }
    props
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
