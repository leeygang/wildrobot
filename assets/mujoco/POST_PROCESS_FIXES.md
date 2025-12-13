# MJCF Post-Processing Fixes

## Summary
Fixed two issues in the generated `wildrobot.xml` MJCF file:
1. ✅ Merged duplicate top-level `<default>` blocks into one
2. ✅ Added explicit `type="sphere"` to collision default geom

## Changes Made to `post_process.py`

### New Function 1: `merge_default_blocks(xml_file)`
Merges multiple top-level `<default>` blocks into a single block.

**Why needed:**
- onshape-to-robot generates multiple top-level `<default>` blocks
- One from base config, one from additional XML includes
- While valid MuJoCo XML, creates redundancy

**What it does:**
- Finds all top-level `<default>` elements
- Keeps first block, merges all children from other blocks into it
- Removes now-empty duplicate blocks

### New Function 2: `fix_collision_default_geom(xml_file)`
Adds explicit `type="sphere"` to collision default geom.

**Why needed:**
- Collision default had `<geom group="3"/>` without type
- Isaac Sim warning: "Could not determine geometry type, defaulting to Sphere"
- Generates warnings during USD conversion

**What it does:**
- Finds collision default class
- Adds `type="sphere"` attribute to geom (matches MuJoCo's implicit default)
- Silences Isaac Sim geometry type warnings

## Usage

```bash
cd assets/mjcf
python3 post_process.py
```

Output:
```
start post process...
Merged 2 default blocks into one
Fixed collision default geom: added type='sphere'
Post process completed
```

## Verification

### Before:
```xml
<default>
  <default class="WildRobotDev">
    ...
    <default class="collision">
      <geom group="3"/>  <!-- No type attribute -->
    </default>
  </default>
</default>
<!-- Additional joints_properties.xml -->
<default>  <!-- DUPLICATE TOP-LEVEL BLOCK -->
  <default class="htd45hServo">
    ...
  </default>
</default>
```

### After:
```xml
<default>
  <default class="WildRobotDev">
    ...
    <default class="collision">
      <geom group="3" type="sphere" />  <!-- Explicit type -->
    </default>
  </default>
  <default class="htd45hServo">  <!-- MERGED INTO SINGLE BLOCK -->
    ...
  </default>
</default>
```

## Isaac Sim USD Conversion

### Fixed Warnings:
- ✅ "Could not determine geometry type, defaulting to Sphere" - FIXED

### Remaining Warnings (Not MJCF Issues):
- ⚠️ "getAttributeCount called on non-existent path /wildrobot/waist/waist/visuals/waist_trunk_support_3"
  - This is a USD scenegraph path issue
  - The geom exists in MJCF as a visual mesh
  - Related to how USD paths are constructed from MJCF hierarchy
  - Does not affect MuJoCo simulation functionality

## Workflow

1. **Generate MJCF from OnShape:**
   ```bash
   onshape-to-robot config.json
   ```

2. **Run post-processing:**
   ```bash
   cd assets/mjcf
   python3 post_process.py
   ```

3. **Convert to USD:**
   ```bash
   bash scripts/convert_mjcf_to_usd
   ```

## Notes

- Both functions are **idempotent** - safe to run multiple times
- Changes are applied in-place to `wildrobot.xml`
- Original formatting is preserved using `ET.indent()`
- Functions only modify what needs to be fixed
