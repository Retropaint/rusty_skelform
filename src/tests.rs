use crate::*;

#[cfg(test)]
mod tests {
    use crate::*;

    fn setup_armature() -> Armature {
        let mut armature = Armature::default();

        #[rustfmt::skip] {
            armature.bones.push(Bone { id: 0, pos: Vec2::new(0.,   150.), ..Default::default() });
            armature.bones.push(Bone { id: 1, pos: Vec2::new(0.,   0.  ), ..Default::default() });
            armature.bones.push(Bone { id: 2, pos: Vec2::new(50.,  0.  ), ..Default::default() });   
            armature.bones.push(Bone { id: 3, pos: Vec2::new(100., 0.  ), ..Default::default() });
        };

        armature
    }

    #[test]
    fn test_animate() {
        let mut armature = setup_armature();
        //crate::animate(&mut armature, 0, 0, false);
    }
}
