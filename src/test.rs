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

        armature.ik_families.push(IkFamily {
            target_id: 0,
            constraint: crate::JointConstraint::None,
            bone_ids: vec![1, 2, 3],
        });

        armature
    }

    #[test]
    fn test_fabrik() {
        let mut armature = setup_armature();

        println!("Target:\n{}\n", armature.bones[0].pos);

        println!("Initial bone positions:");
        println!("{}", armature.bones[1].pos);
        println!("{}", armature.bones[2].pos);
        println!("{}", armature.bones[3].pos);
        println!();

        let start_pos = armature.bones[armature.ik_families[0].bone_ids[0] as usize].pos;

        println!("Forward reaching:");
        crate::test::forward_reaching(&armature.ik_families[0], &mut armature.bones);
        println!();

        println!("Backward-reaching:");
        crate::test::backward_reaching(&armature.ik_families[0], &mut armature.bones, start_pos);
        println!();
    }
}

pub fn forward_reaching(family: &IkFamily, bones: &mut Vec<Bone>) {
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

        println!("{}", bone!().pos);

        next_pos = bone!().pos;
    }
}

pub fn backward_reaching(family: &IkFamily, bones: &mut Vec<Bone>, start_pos: Vec2) {
    let mut prev_pos = start_pos;
    let mut prev_length = 0.;
    for i in 0..family.bone_ids.len() {
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

        println!("{}", bone!().pos);

        prev_pos = bone!().pos;
    }
}
