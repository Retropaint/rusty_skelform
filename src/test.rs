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
            constraint: crate::JointConstraint::CounterClockwise,
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
        crate::test::forward_reaching(&armature.ik_families[0], &mut armature.bones, start_pos);
        println!();

        let constraint_str = match armature.ik_families[0].constraint  {
            JointConstraint::Clockwise => " (Clockwise)",
            JointConstraint::CounterClockwise => " (Clockwise)",
            _ => "",
        };
        println!("Backward-reaching{}:", constraint_str);
        crate::test::backward_reaching(&armature.ik_families[0], &mut armature.bones, start_pos);
        println!();

        println!("Rotations:");
        crate::test::get_ik_rots(&armature.ik_families[0], &mut armature.bones);
        println!();
    }
}

pub fn forward_reaching(family: &IkFamily, bones: &mut Vec<Bone>, start_pos: Vec2) {
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

    let base_line = normalize(bones[family.target_id as usize].pos - start_pos);
    let base_angle = base_line.y.atan2(base_line.x);

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

        if i != 0 && i != family.bone_ids.len() - 1 && family.constraint != JointConstraint::None {
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

        println!("{}", bone!().pos);

        prev_pos = bone!().pos;
    }
}

pub fn get_ik_rots(family: &IkFamily, bones: &mut Vec<Bone>) {
    let end_bone = &bones[*family.bone_ids.last().unwrap() as usize];
    let mut tip_pos = end_bone.pos;
    for i in (0..family.bone_ids.len()).rev() {
        if family.target_id == -1 || i == family.bone_ids.len() - 1 {
            continue;
        }
        macro_rules! bone {
            () => {
                bones[family.bone_ids[i] as usize]
            };
        }

        let dir = tip_pos - bone!().pos;
        bone!().rot = dir.y.atan2(dir.x);

        println!("{} ({})", bone!().rot, bone!().rot * 180. / 3.14);

        tip_pos = bone!().pos;
    }
}
