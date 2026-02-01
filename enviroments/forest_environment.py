"""
Forest Environment with 10 Trees for Drone Navigation Simulation
Creates a simulation environment with randomly placed procedural trees.

Usage (FROM IsaacLab directory):
    cd ~/projects/isaac/IsaacLab
    ./isaaclab.sh -p ~/Desktop/Lasitha/drone_rl_project/scripts/forest_environment.py
"""

import argparse
import random
import math

from isaaclab.app import AppLauncher

# Configuration
NUM_TREES = 10
FOREST_SIZE = 30.0  # meters - area where trees will be placed
MIN_TREE_SPACING = 4.0  # minimum distance between trees
RANDOM_SEED = 42  # for reproducible layouts

# Argument parser
parser = argparse.ArgumentParser(description="Forest Environment for Drone Navigation")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Start Isaac Sim with GUI
print(f"\nStarting Isaac Sim...")
print(f"   (This may take 30-60 seconds...)\n")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print(f"Loading forest environment with {NUM_TREES} trees...\n")

import isaaclab.sim as sim_utils
from pxr import UsdGeom, Gf, UsdLux, UsdShade, Sdf
import omni.usd

# Create simulation context
sim_cfg = sim_utils.SimulationCfg(dt=0.01)
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view([0.0, -40.0, 20.0], [0.0, 0.0, 0.0])

# Get stage
stage = omni.usd.get_context().get_stage()


def create_material(stage, mat_path, color):
    """Create a UsdPreviewSurface material visible in RTX renderer.

    displayColor only works in Storm/HydraGL. Isaac Sim uses RTX by default,
    which requires proper UsdShade materials to render geometry.
    """
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def generate_tree_positions(num_trees, area_size, min_spacing, seed=None):
    """Generate random non-overlapping positions for trees."""
    if seed is not None:
        random.seed(seed)

    positions = []
    max_attempts = 1000
    half_size = area_size / 2

    for _ in range(num_trees):
        for attempt in range(max_attempts):
            # Generate random position
            x = random.uniform(-half_size, half_size)
            y = random.uniform(-half_size, half_size)

            # Check distance from all existing trees
            valid = True
            for px, py in positions:
                dist = math.sqrt((x - px)**2 + (y - py)**2)
                if dist < min_spacing:
                    valid = False
                    break

            if valid:
                positions.append((x, y))
                break
        else:
            print(f"   Warning: Could not place tree {len(positions)+1} with minimum spacing")
            x = random.uniform(-half_size, half_size)
            y = random.uniform(-half_size, half_size)
            positions.append((x, y))

    return positions


def add_tree_procedural(stage, prim_path, position, rotation=0, scale=1.0,
                        tree_type="pine", trunk_material=None, canopy_material=None):
    """Add a procedural tree at the specified position.

    Tree types:
        - pine: cylinder trunk + cone canopy
        - oak: thicker trunk + sphere canopy
        - palm: tall thin trunk + flattened sphere crown
    """
    # Create Xform container using direct USD API (more reliable than omni.kit.commands)
    tree_xform = UsdGeom.Xform.Define(stage, prim_path)
    tree_prim = tree_xform.GetPrim()

    # Position the tree
    translate_op = tree_xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(position[0], position[1], 0))

    # Scale
    scale_op = tree_xform.AddScaleOp()
    scale_op.Set(Gf.Vec3d(scale, scale, scale))

    if tree_type == "pine":
        # Pine tree: brown cylinder trunk + green cone canopy
        trunk_height = 4.0
        trunk_radius = 0.3
        canopy_height = 6.0
        canopy_radius = 2.0

        # Create trunk
        trunk = UsdGeom.Cylinder.Define(stage, f"{prim_path}/Trunk")
        trunk.CreateHeightAttr(trunk_height)
        trunk.CreateRadiusAttr(trunk_radius)
        trunk_xform = UsdGeom.Xformable(trunk.GetPrim())
        trunk_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, trunk_height / 2))
        if trunk_material:
            UsdShade.MaterialBindingAPI(trunk.GetPrim()).Bind(trunk_material)

        # Create canopy (cone)
        canopy = UsdGeom.Cone.Define(stage, f"{prim_path}/Canopy")
        canopy.CreateHeightAttr(canopy_height)
        canopy.CreateRadiusAttr(canopy_radius)
        canopy_xform = UsdGeom.Xformable(canopy.GetPrim())
        canopy_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, trunk_height + canopy_height / 2))
        if canopy_material:
            UsdShade.MaterialBindingAPI(canopy.GetPrim()).Bind(canopy_material)

    elif tree_type == "oak":
        # Oak tree: thicker trunk + sphere canopy
        trunk_height = 3.0
        trunk_radius = 0.5
        canopy_radius = 3.0

        # Create trunk
        trunk = UsdGeom.Cylinder.Define(stage, f"{prim_path}/Trunk")
        trunk.CreateHeightAttr(trunk_height)
        trunk.CreateRadiusAttr(trunk_radius)
        trunk_xform = UsdGeom.Xformable(trunk.GetPrim())
        trunk_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, trunk_height / 2))
        if trunk_material:
            UsdShade.MaterialBindingAPI(trunk.GetPrim()).Bind(trunk_material)

        # Create canopy (sphere)
        canopy = UsdGeom.Sphere.Define(stage, f"{prim_path}/Canopy")
        canopy.CreateRadiusAttr(canopy_radius)
        canopy_xform = UsdGeom.Xformable(canopy.GetPrim())
        canopy_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, trunk_height + canopy_radius * 0.8))
        if canopy_material:
            UsdShade.MaterialBindingAPI(canopy.GetPrim()).Bind(canopy_material)

    elif tree_type == "palm":
        # Palm tree: tall thin trunk + flattened sphere crown
        trunk_height = 8.0
        trunk_radius = 0.25

        # Create trunk
        trunk = UsdGeom.Cylinder.Define(stage, f"{prim_path}/Trunk")
        trunk.CreateHeightAttr(trunk_height)
        trunk.CreateRadiusAttr(trunk_radius)
        trunk_xform = UsdGeom.Xformable(trunk.GetPrim())
        trunk_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, trunk_height / 2))
        if trunk_material:
            UsdShade.MaterialBindingAPI(trunk.GetPrim()).Bind(trunk_material)

        # Create crown (flattened sphere)
        crown = UsdGeom.Sphere.Define(stage, f"{prim_path}/Canopy")
        crown.CreateRadiusAttr(1.5)
        crown_xform = UsdGeom.Xformable(crown.GetPrim())
        crown_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, trunk_height + 0.5))
        crown_xform.AddScaleOp().Set(Gf.Vec3d(1.5, 1.5, 0.5))
        if canopy_material:
            UsdShade.MaterialBindingAPI(crown.GetPrim()).Bind(canopy_material)

    return tree_prim


# Add lighting
print("Setting up lighting...")
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(1500)
xform = UsdGeom.Xformable(sun)
xform.AddRotateXOp().Set(-45)
xform.AddRotateYOp().Set(30)

dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(500)

# Create forest container using direct USD API
UsdGeom.Xform.Define(stage, "/World/Forest")

# Create shared materials (RTX renderer requires UsdShade materials, not displayColor)
print("Creating materials...")
materials_path = "/World/Materials"
UsdGeom.Xform.Define(stage, materials_path)

# Trunk materials (brown variants)
mat_trunk_pine = create_material(stage, f"{materials_path}/TrunkPine", (0.4, 0.25, 0.1))
mat_trunk_oak = create_material(stage, f"{materials_path}/TrunkOak", (0.35, 0.2, 0.1))
mat_trunk_palm = create_material(stage, f"{materials_path}/TrunkPalm", (0.5, 0.35, 0.2))

# Canopy materials (green variants)
mat_canopy_pine = create_material(stage, f"{materials_path}/CanopyPine", (0.1, 0.5, 0.15))
mat_canopy_oak = create_material(stage, f"{materials_path}/CanopyOak", (0.2, 0.6, 0.2))
mat_canopy_palm = create_material(stage, f"{materials_path}/CanopyPalm", (0.15, 0.55, 0.15))

TRUNK_MATERIALS = {
    "pine": mat_trunk_pine,
    "oak": mat_trunk_oak,
    "palm": mat_trunk_palm,
}
CANOPY_MATERIALS = {
    "pine": mat_canopy_pine,
    "oak": mat_canopy_oak,
    "palm": mat_canopy_palm,
}

# Generate tree positions
print(f"Generating {NUM_TREES} tree positions...")
tree_positions = generate_tree_positions(
    NUM_TREES,
    FOREST_SIZE,
    MIN_TREE_SPACING,
    RANDOM_SEED
)

# Place trees with random types
print(f"Placing trees...")
TREE_TYPES = ["pine", "oak", "palm"]
trees = []
tree_type_counts = {"pine": 0, "oak": 0, "palm": 0}

for i, (x, y) in enumerate(tree_positions):
    prim_path = f"/World/Forest/Tree_{i:02d}"

    # Random tree type and scale variation
    tree_type = random.choice(TREE_TYPES)
    scale = random.uniform(0.8, 1.2)

    try:
        tree_prim = add_tree_procedural(
            stage, prim_path, (x, y), 0, scale, tree_type,
            trunk_material=TRUNK_MATERIALS[tree_type],
            canopy_material=CANOPY_MATERIALS[tree_type],
        )
        trees.append(tree_prim)
        tree_type_counts[tree_type] += 1
        print(f"   Tree {i+1:2d}: {tree_type:5s} at ({x:6.2f}, {y:6.2f}), scale {scale:.2f}")
    except Exception as e:
        print(f"   Failed to place tree {i+1}: {e}")

print(f"\nSuccessfully placed {len(trees)} trees!\n")

# Add ground plane
print("Adding ground plane...")
ground_size = FOREST_SIZE * 1.5
ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
ground.CreatePointsAttr([
    Gf.Vec3f(-ground_size, -ground_size, 0),
    Gf.Vec3f(ground_size, -ground_size, 0),
    Gf.Vec3f(ground_size, ground_size, 0),
    Gf.Vec3f(-ground_size, ground_size, 0)
])
ground.CreateFaceVertexCountsAttr([4])
ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
mat_ground = create_material(stage, f"{materials_path}/Ground", (0.3, 0.5, 0.2))
UsdShade.MaterialBindingAPI(ground.GetPrim()).Bind(mat_ground)

# Add boundary markers
print("Adding boundary markers...")
boundary_prim_path = "/World/Boundaries"
UsdGeom.Xform.Define(stage, boundary_prim_path)
mat_marker = create_material(stage, f"{materials_path}/BoundaryMarker", (1.0, 0.3, 0.3))

# Corner markers
half_size = FOREST_SIZE / 2
corners = [
    (-half_size, -half_size),
    (half_size, -half_size),
    (half_size, half_size),
    (-half_size, half_size)
]

for i, (cx, cy) in enumerate(corners):
    marker_path = f"{boundary_prim_path}/Corner_{i}"
    marker = UsdGeom.Cylinder.Define(stage, marker_path)
    marker.CreateRadiusAttr(0.3)
    marker.CreateHeightAttr(3.0)

    xformable = UsdGeom.Xformable(marker.GetPrim())
    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(cx, cy, 1.5))
    UsdShade.MaterialBindingAPI(marker.GetPrim()).Bind(mat_marker)

# Add start position marker (green sphere)
start_marker = UsdGeom.Sphere.Define(stage, "/World/StartPosition")
start_marker.CreateRadiusAttr(0.5)
mat_start = create_material(stage, f"{materials_path}/StartMarker", (0.2, 1.0, 0.2))
UsdShade.MaterialBindingAPI(start_marker.GetPrim()).Bind(mat_start)
xformable = UsdGeom.Xformable(start_marker.GetPrim())
translate_op = xformable.AddTranslateOp()
translate_op.Set(Gf.Vec3d(-half_size + 2, 0, 2))

# Add goal position marker (blue sphere)
goal_marker = UsdGeom.Sphere.Define(stage, "/World/GoalPosition")
goal_marker.CreateRadiusAttr(0.5)
mat_goal = create_material(stage, f"{materials_path}/GoalMarker", (0.2, 0.2, 1.0))
UsdShade.MaterialBindingAPI(goal_marker.GetPrim()).Bind(mat_goal)
xformable = UsdGeom.Xformable(goal_marker.GetPrim())
translate_op = xformable.AddTranslateOp()
translate_op.Set(Gf.Vec3d(half_size - 2, 0, 2))

# Print environment info
print("\n" + "="*70)
print("FOREST ENVIRONMENT SUMMARY")
print("="*70)
print(f"   Tree types:      Pine (cone), Oak (sphere), Palm (flat crown)")
print(f"   Number of trees: {NUM_TREES}")
print(f"     - Pine trees:  {tree_type_counts['pine']}")
print(f"     - Oak trees:   {tree_type_counts['oak']}")
print(f"     - Palm trees:  {tree_type_counts['palm']}")
print(f"   Forest area:     {FOREST_SIZE}m x {FOREST_SIZE}m")
print(f"   Min tree spacing: {MIN_TREE_SPACING}m")
print()
print("   Markers:")
print("     - Red cylinders:  Boundary corners")
print("     - Green sphere:   Start position")
print("     - Blue sphere:    Goal position")

print("\n" + "="*70)
print("ISAAC SIM CONTROLS:")
print("="*70)
print("   Left Mouse + Drag:     Rotate view")
print("   Right Mouse + Drag:    Pan view")
print("   Middle Mouse / Scroll: Zoom in/out")
print("   WASD:                  Move camera")
print("   Q/E:                   Up/Down")
print("   Double-click object:   Focus on it")
print("   F:                     Frame selected object")
print("\n   Press Ctrl+C in this terminal to exit")
print("="*70 + "\n")

# Reset and run
sim.reset()
print("[INFO]: Setup complete...")

# Keep window open
try:
    frame_count = 0
    while simulation_app.is_running():
        sim.step()
        frame_count += 1

        # Print a message every 300 frames (~10 seconds)
        if frame_count % 300 == 0:
            print(f"   Running... (Frame {frame_count})")

except KeyboardInterrupt:
    print("\n\nClosing Isaac Sim...")

simulation_app.close()
print("Done!\n")
