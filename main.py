import os
import json
import time
from pathlib import Path
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box

def get_face_features(face):
    """Extracts a 4-dimensional geometric feature vector from a CAD face."""
    # 1. Calculate Area & Center of Mass
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    area = props.Mass()
    cog = props.CentreOfMass()
    
    # 2. Identify Surface Type
    surface = BRepAdaptor_Surface(face)
    surf_type = surface.GetType()
    if surf_type == GeomAbs_Plane:
        type_code = 0.0
    elif surf_type == GeomAbs_Cylinder:
        type_code = 1.0
    else:
        type_code = 2.0  # Cones, Spheres, Toruses, or complex B-Splines
        
    # 3. Calculate Bounding Box Volume (Aspect Ratio Descriptor)
    bbox = Bnd_Box()
    brepbndlib.Add(face, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    bbox_volume = dx * dy * dz
    
    # Return a flat list of features
    return [round(area, 4), type_code, round(cog.X(), 4), round(bbox_volume, 4)]

def extract_part_graph(file_path):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(file_path)) != 1:
        return None
    
    reader.TransferRoots()
    shape = reader.OneShape()
    
    face_to_id = {}
    nodes = []
    
    f_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    idx = 0
    while f_explorer.More():
        face = f_explorer.Current()
        face_to_id[face] = idx
        
        # Pull our advanced 4D feature array
        features = get_face_features(face)
        nodes.append({
            "node_id": idx,
            "features": features
        })
        
        idx += 1
        f_explorer.Next()
        
    edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces)
    
    links = []
    for i in range(1, edge_to_faces.Size() + 1):
        faces_list = edge_to_faces.FindFromIndex(i)
        if faces_list.Size() == 2:
            id_a = face_to_id.get(faces_list.First())
            id_b = face_to_id.get(faces_list.Last())
            if id_a is not None and id_b is not None:
                connection = sorted([id_a, id_b])
                link_entry = {"source": connection[0], "target": connection[1]}
                if link_entry not in links:
                    links.append(link_entry)
                    
    return {"nodes": nodes, "links": links}

# --- Main Dataset Generation ---
base_path = Path("data_chunks/abc_0000_step_v00")
# Let's scale up to 100 parts now to give our network real data variety!
step_files = list(base_path.rglob("*.step"))[:100]  

if not step_files:
    print(f"❌ No .step files found in: {base_path}")
else:
    print(f"📦 Found {len(step_files)} files. Extracting 4D geometric features...")
    start_time = time.time()
    
    dataset = {}
    for file_path in step_files:
        try:
            graph_data = extract_part_graph(file_path)
            if graph_data and len(graph_data["nodes"]) > 0:
                dataset[file_path.name] = graph_data
        except Exception as e:
            continue  # Skip any anomalous or corrupted dataset files silently
            
    with open("cad_graph_dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"✅ Enhanced Dataset generated in {round(time.time() - start_time, 2)} seconds!")