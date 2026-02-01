"""
View a procedural tree in Isaac Sim (no external assets needed)
Usage (FROM IsaacLab directory):
    cd ~/projects/isaac/IsaacLab
    ./isaaclab.sh -p ~/Desktop/Lasitha/drone_rl_project/scripts/view_tree_isaac.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View a procedural tree in Isaac Sim")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print("\nStarting Isaac Sim...")
print("   (This may take 30-60 seconds...)\n")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("Creating procedural tree...\n")

import isaaclab.sim as sim_utils
from pxr import UsdGeom, Gf, UsdLux
import omni.kit.commands
import omni.usd

# Create simulation context
sim_cfg = sim_utils.SimulationCfg(dt=0.01)
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view([8.0, 8.0, 6.0], [0.0, 0.0, 3.0])

stage = omni.usd.get_context().get_stage()

# Add lighting
print("Setting up lighting...")
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(1000)
xform = UsdGeom.Xformable(sun)
xform.AddRotateXOp().Set(-45)

dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(300)


def create_procedural_tree(stage, prim_path, tree_type="pine"):
    """Create a procedural tree using Isaac Sim primitives."""

    omni.kit.commands.execute(
        "CreatePrimWithDefaultXform",
        prim_type="Xform",
        prim_path=prim_path,
    )

    tree_prim = stage.GetPrimAtPath(prim_path)

    if tree_type == "pine":
        # Pine tree: brown cylinder trunk + green cone canopy
        trunk_height = 4.0
        trunk_radius = 0.3
        canopy_height = 6.0
        canopy_radius = 2.5

        # Trunk
        trunk = UsdGeom.Cylinder.Define(stage, f"{prim_path}/Trunk")
        trunk.CreateHeightAttr(trunk_height)
        trunk.CreateRadiusAttr(trunk_radius)
        trunk.CreateDisplayColorAttr([(0.4, 0.25, 0.1)])  # Brown

        trunk_xform = UsdGeom.Xformable(trunk)
        trunk_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, trunk_height / 2))

        # Canopy (cone)
        canopy = UsdGeom.Cone.Define(stage, f"{prim_path}/Canopy")
        canopy.CreateHeightAttr(canopy_height)
        canopy.CreateRadiusAttr(canopy_radius)
        canopy.CreateDisplayColorAttr([(0.1, 0.5, 0.15)])  # Dark green

        canopy_xform = UsdGeom.Xformable(canopy)
        canopy_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, trunk_height + canopy_height / 2))

    elif tree_type == "oak":
        # Oak tree: thicker trunk + sphere canopy
        trunk_height = 3.0
        trunk_radius = 0.5
        canopy_radius = 3.0

        # Trunk
        trunk = UsdGeom.Cylinder.Define(stage, f"{prim_path}/Trunk")
        trunk.CreateHeightAttr(trunk_height)
        trunk.CreateRadiusAttr(trunk_radius)
        trunk.CreateDisplayColorAttr([(0.35, 0.2, 0.1)])  # Dark brown

        trunk_xform = UsdGeom.Xformable(trunk)
        trunk_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, trunk_height / 2))

        # Canopy (sphere)
        canopy = UsdGeom.Sphere.Define(stage, f"{prim_path}/Canopy")
        canopy.CreateRadiusAttr(canopy_radius)
        canopy.CreateDisplayColorAttr([(0.2, 0.6, 0.2)])  # Green

        canopy_xform = UsdGeom.Xformable(canopy)
        canopy_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, trunk_height + canopy_radius * 0.8))

    elif tree_type == "palm":
        # Palm tree: tall thin trunk + small top
        trunk_height = 8.0
        trunk_radius = 0.25

        # Trunk
        trunk = UsdGeom.Cylinder.Define(stage, f"{prim_path}/Trunk")
        trunk.CreateHeightAttr(trunk_height)
        trunk.CreateRadiusAttr(trunk_radius)
        trunk.CreateDisplayColorAttr([(0.5, 0.35, 0.2)])  # Light brown

        trunk_xform = UsdGeom.Xformable(trunk)
        trunk_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, trunk_height / 2))

        # Crown (flattened sphere)
        crown = UsdGeom.Sphere.Define(stage, f"{prim_path}/Crown")
        crown.CreateRadiusAttr(1.5)
        crown.CreateDisplayColorAttr([(0.15, 0.55, 0.15)])  # Green

        crown_xform = UsdGeom.Xformable(crown)
        crown_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, trunk_height + 0.5))
        crown_xform.AddScaleOp().Set(Gf.Vec3d(1.5, 1.5, 0.5))

    return tree_prim


# Create multiple tree types for viewing
print("Creating trees...")

tree_types = ["pine", "oak", "palm"]
positions = [(-4, 0), (0, 0), (4, 0)]

for i, (tree_type, pos) in enumerate(zip(tree_types, positions)):
    prim_path = f"/World/Tree_{tree_type}"
    tree = create_procedural_tree(stage, prim_path, tree_type)

    # Position the tree
    tree_xform = UsdGeom.Xformable(tree)
    tree_xform.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], 0))

    print(f"   Created {tree_type} tree at position {pos}")

# Add ground plane
print("Adding ground plane...")
ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
size = 15
ground.CreatePointsAttr([
    Gf.Vec3f(-size, -size, 0),
    Gf.Vec3f(size, -size, 0),
    Gf.Vec3f(size, size, 0),
    Gf.Vec3f(-size, size, 0)
])
ground.CreateFaceVertexCountsAttr([4])
ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
ground.CreateDisplayColorAttr([(0.3, 0.5, 0.25)])  # Grass green

# Add labels
print("\n" + "="*60)
print("PROCEDURAL TREES (using Isaac Sim built-in primitives)")
print("="*60)
print("   Left:   Pine tree  (cylinder + cone)")
print("   Center: Oak tree   (cylinder + sphere)")
print("   Right:  Palm tree  (cylinder + flattened sphere)")
print()
print("   These use NO external assets - just USD primitives!")
print("="*60)

print("\nISAAC SIM CONTROLS:")
print("   Left Mouse + Drag:  Rotate view")
print("   Right Mouse + Drag: Pan view")
print("   Scroll:             Zoom")
print("   WASD / Q/E:         Move camera")
print("\n   Press Ctrl+C to exit")
print("="*60 + "\n")

# Reset and run
sim.reset()
print("[INFO]: Setup complete...")

try:
    while simulation_app.is_running():
        sim.step()
except KeyboardInterrupt:
    print("\n\nClosing Isaac Sim...")

simulation_app.close()
print("Done!\n")
