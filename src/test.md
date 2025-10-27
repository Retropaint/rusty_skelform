1. Add all bone ids and element ids that are affected in animations(s)
2. Attempt to reset everything, but ignoring those in step 1

```rust
let affected = []
for anim in animations {
  affected = animate()
}

for bone in bones {
  macro attempt (bid, eid) {
    if !affected.contains(bid, eid) {
      bone = resetInterp()
    }
  }

  attempt(bone.id, 0) // posX
  attempt(bone.id, 1) // posY
  attempt(bone.id, 2) // rot
  attempt(bone.id, 3) // scaleX
  attempt(bone.id, 4) // scaleY
}
```
