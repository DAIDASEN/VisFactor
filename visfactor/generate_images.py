from utils.CF1 import *
from utils.CF2 import *
from utils.CF3 import *
from utils.CS1 import *
from utils.CS2 import *
from utils.CS3 import *
from utils.MA1 import *
from utils.S1 import *
from utils.S2 import *
from utils.SS3 import *
from utils.VZ1 import *
from utils.VZ2 import *
import os
import cv2
import random

data_meta = "G_N_Data.csv"
image_meta = "G_N_Images"

CF1_config = {
    "num": 32,
    "rows": 6,
    "cols": 6,
    "density": 0.2,
    "model_0": "0,1-0,2; 0,2-0,3; 0,3-1,2; 1,2-2,2; 2,2-2,1; 2,1-1,0; 1,0-0,1",
    "model_1": "0,0-0,1; 0,1-0,2; 0,2-0,3; 0,3-1,2; 1,2-2,1; 2,1-1,1; 1,1-0,0",
    "model_2": "1,0-1,1; 1,1-0,2; 0,2-0,3; 0,3-1,3; 1,3-2,2; 2,2-2,1; 2,1-1,0",
    "model_3": "2,0-1,1; 1,1-0,2; 0,2-1,3; 1,3-2,3; 2,3-2,2; 2,2-2,1; 2,1-2,0",
    "model_4": "2,0-1,1; 1,1-0,1; 0,1-1,2; 1,2-1,3; 1,3-2,2; 2,2-2,1; 2,1-2,0",
}
CF2_config = {
    "num": 80,
    "rows": 3,
    "cols": 3,
    "density": 0.2,
    "model": "0,2-0,1; 0,1-1,1; 1,1-2,0; 1,1-2,2"
}
CF3_config = {
    "num": 64,
    "rows": 5,
    "cols": 5,
    "min_steps": 4,
    "max_steps": 4,
}
CS1_config = {
    "num": 20,
    "input_dir": "Collected_Figures",
    "severity": 0.4
}
CS2_config = {
    "num": 50,
    "min_length": 3,
    "max_length": 6,
    "severity": 0.4
}
CS3_config = {
    "num": 24,
    "input_dir": "Collected_Figures",
    "severity": 0.4
}
MA1_config = {
    "num": 42,
    "input_dir": "Collected_Figures",
    "capacity": 21,
}
S1_config = {
    "num": 20,
    "min_vertex": 4,
    "max_vertex": 8,
}
S2_config = {
    "num": 42
}
SS3_config = {
    "num": 40,
    "rows": 7,
    "cols": 8,
    "blocked_ratio": 0.25,
    "num_buildings": 10
}
VZ1_config = {
    "num": 48,
    "model_0": "0,1-0,3; 0,3-1,3; 1,3-1,4; 1,4-3,4; 3,4-3,3; 3,3-4,3; 4,3-4,1; 4,1-3,1; 3,1-3,0; 3,0-1,0; 1,0-1,1; 1,1-0,1",
    "model_1": "0,1-0,3; 0,3-2,4; 2,4-4,3; 4,3-4,1; 4,1-2,0; 2,0-0,1",
    "model_2": "0,0-0,5; 0,5-5,5; 5,5-5,0; 5,0-0,0",
    "model_3": "4,0-4,4; 4,4-0,2; 0,2-4,0",
}
VZ2_config = {
    "num": 20,
    "n": 3,
    "min_steps": 1,
    "max_steps": 3,
}

with open(data_meta, "w") as f:
    f.write(f"subtests,eval_index,questions,images,answers\n")

os.mkdir(image_meta)
os.mkdir(f"{image_meta}/CF1-Hidden-Figures-Test")
os.mkdir(f"{image_meta}/CF2-Hidden-Patterns-Test")
os.mkdir(f"{image_meta}/CF3-Copying-Test")
os.mkdir(f"{image_meta}/CS1-Gestalt-Completion-Test")
os.mkdir(f"{image_meta}/CS2-Concealed-Words-Test")
os.mkdir(f"{image_meta}/CS3-Snowy-Pictures")
os.mkdir(f"{image_meta}/MA1-Figure-Number-Test")
os.mkdir(f"{image_meta}/S1-Card-Rotations-Test")
os.mkdir(f"{image_meta}/S2-Cube-Comparisons-Test")
os.mkdir(f"{image_meta}/SS3-Map-Planning-Test")
os.mkdir(f"{image_meta}/VZ1-Form-Board-Test")
os.mkdir(f"{image_meta}/VZ2-Paper-Folding-Test")

print(">>>>Initialization completed. Starting CF1-Hidden-Figures-Test ...")

grid = Grid(rows=CF1_config["rows"], cols=CF1_config["cols"])
gen = GridFigureGenerator(grid, density=CF1_config["density"])

model_0_edges = parse_model(CF1_config["model_0"])
grid.draw(model_0_edges, f"{image_meta}/CF1-Hidden-Figures-Test/c-0.png", tight=True)
model_1_edges = parse_model(CF1_config["model_1"])
grid.draw(model_1_edges, f"{image_meta}/CF1-Hidden-Figures-Test/c-1.png", tight=True)
model_2_edges = parse_model(CF1_config["model_2"])
grid.draw(model_2_edges, f"{image_meta}/CF1-Hidden-Figures-Test/c-2.png", tight=True)
model_3_edges = parse_model(CF1_config["model_3"])
grid.draw(model_3_edges, f"{image_meta}/CF1-Hidden-Figures-Test/c-3.png", tight=True)
model_4_edges = parse_model(CF1_config["model_4"])
grid.draw(model_4_edges, f"{image_meta}/CF1-Hidden-Figures-Test/c-4.png", tight=True)

for idx in range(CF1_config["num"]):
    pattern_edges = gen.sample()
    included_0 = model_in_pattern(model_0_edges, pattern_edges)
    included_1 = model_in_pattern(model_1_edges, pattern_edges)
    included_2 = model_in_pattern(model_2_edges, pattern_edges)
    included_3 = model_in_pattern(model_3_edges, pattern_edges)
    included_4 = model_in_pattern(model_4_edges, pattern_edges)
    success = int(included_0) + int(included_1) + int(included_2) + int (included_3) + int(included_4)
    while success < 1 or success > 4:
        pattern_edges = gen.sample()
        included_0 = model_in_pattern(model_0_edges, pattern_edges)
        included_1 = model_in_pattern(model_1_edges, pattern_edges)
        included_2 = model_in_pattern(model_2_edges, pattern_edges)
        included_3 = model_in_pattern(model_3_edges, pattern_edges)
        included_4 = model_in_pattern(model_4_edges, pattern_edges)
        success = int(included_0) + int(included_1) + int(included_2) + int (included_3) + int(included_4)
    grid.draw(pattern_edges, f"{image_meta}/CF1-Hidden-Figures-Test/{idx}.png")
    with open(data_meta, "a") as f:
        f.write(f"CF1,{idx},,{image_meta}/CF1-Hidden-Figures-Test/c-0.png;{image_meta}/CF1-Hidden-Figures-Test/{idx}.png,{'T' if included_0 else 'F'}\n")
        f.write(f"CF1,{idx},,{image_meta}/CF1-Hidden-Figures-Test/c-1.png;{image_meta}/CF1-Hidden-Figures-Test/{idx}.png,{'T' if included_1 else 'F'}\n")
        f.write(f"CF1,{idx},,{image_meta}/CF1-Hidden-Figures-Test/c-2.png;{image_meta}/CF1-Hidden-Figures-Test/{idx}.png,{'T' if included_2 else 'F'}\n")
        f.write(f"CF1,{idx},,{image_meta}/CF1-Hidden-Figures-Test/c-3.png;{image_meta}/CF1-Hidden-Figures-Test/{idx}.png,{'T' if included_3 else 'F'}\n")
        f.write(f"CF1,{idx},,{image_meta}/CF1-Hidden-Figures-Test/c-4.png;{image_meta}/CF1-Hidden-Figures-Test/{idx}.png,{'T' if included_4 else 'F'}\n")

print(">>>>CF1-Hidden-Figures-Test completed. Starting CF2-Hidden-Patterns-Test ...")

gen = GridPatternGenerator(rows=CF2_config["rows"], cols=CF2_config["cols"], density=CF2_config["density"])
model_edges = parse_edges(CF2_config["model"])
gen.draw_edges(model_edges, f"{image_meta}/CF2-Hidden-Patterns-Test/m.png")

eval_idx = []
for i in range(CF2_config["num"]):
    eval_idx.extend([i] * 5)
random.shuffle(eval_idx)

for idx in range(CF2_config["num"] * 5):
    pattern_edges = gen.generate_pattern()
    contains = gen.contains_model(pattern_edges, model_edges)
    if idx < int(CF2_config["num"] * 2.5):
        while not contains:
            pattern_edges = gen.generate_pattern()
            contains = gen.contains_model(pattern_edges, model_edges)
    else:
        while contains:
            pattern_edges = gen.generate_pattern()
            contains = gen.contains_model(pattern_edges, model_edges)
    img_path = f"{image_meta}/CF2-Hidden-Patterns-Test/{idx}.png"
    gen.draw_edges(pattern_edges, img_path)
    with open(data_meta, "a") as f:
        f.write(f"CF2,{eval_idx[idx]},,{image_meta}/CF2-Hidden-Patterns-Test/m.png;{image_meta}/CF2-Hidden-Patterns-Test/{idx}.png,{'T' if contains else 'F'}\n")

print(">>>>CF2-Hidden-Patterns-Test completed. Starting CF3-Copying-Test ...")

grid = GridConfig(rows=CF3_config["rows"], cols=CF3_config["cols"])
gen = GridWalkGenerator(grid, CF3_config["min_steps"], CF3_config["max_steps"])
for idx in range(CF3_config["num"]):
    end = gen.generate_pair(idx, f"{image_meta}/CF3-Copying-Test/{idx}.png")
    with open(data_meta, "a") as f:
        f.write(f"CF3,{idx},,{image_meta}/CF3-Copying-Test/{idx}-0.png;{image_meta}/CF3-Copying-Test/{idx}-1.png,\"{end}\"\n")

print(">>>>CF3-Copying-Test completed. Starting CS1-Gestalt-Completion-Test ...")

CS1_dataset = MaskedImageDataset(CS1_config["input_dir"], severity=CS1_config["severity"])
for idx, (img, label) in enumerate(CS1_dataset):
    if idx == CS1_config["num"]:
        break
    img.save(f"{image_meta}/CS1-Gestalt-Completion-Test/{idx}.png")
    with open(data_meta, "a") as f:
        f.write(f"CS1,{idx},,{image_meta}/CS1-Gestalt-Completion-Test/{idx}.png,{label}\n")

print(">>>>CS1-Gestalt-Completion-Test completed. Starting CS2-Concealed-Words-Test ...")

CS2_dataset = WordImageDataset(CS2_config["min_length"], CS2_config["max_length"], severity=CS2_config["severity"])
for idx, (img, label) in enumerate(CS2_dataset):
    if idx == CS2_config["num"]:
        break
    cv2.imwrite(f"{image_meta}/CS2-Concealed-Words-Test/{idx}.png", img)
    with open(data_meta, "a") as f:
        f.write(f"CS2,{idx},,{image_meta}/CS2-Concealed-Words-Test/{idx}.png,{label}\n")

print(">>>>CS2-Concealed-Words-Test completed. Starting CS3-Snowy-Pictures ...")

CS3_dataset = NoisyImageDataset(root_dir=CS3_config["input_dir"], severity=CS3_config["severity"],)
for idx, (img, label) in enumerate(CS3_dataset):
    if idx == CS3_config["num"]:
        break
    img.save(f"{image_meta}/CS3-Snowy-Pictures/{idx}.png")
    with open(data_meta, "a") as f:
        f.write(f"CS3,{idx},,{image_meta}/CS3-Snowy-Pictures/{idx}.png,{label}\n")

print(">>>>CS3-Snowy-Pictures completed. Starting MA1-Figure-Number-Test ...")

MA1_dataset = CompositeGridDataset(MA1_config["input_dir"], capacity=MA1_config["capacity"])
for idx, (big_img, obj_img, label) in enumerate(MA1_dataset):
    if idx == MA1_config["num"]:
        break
    big_img.save(f"{image_meta}/MA1-Figure-Number-Test/{idx}-0.png")
    obj_img.save(f"{image_meta}/MA1-Figure-Number-Test/{idx}-1.png")
    with open(data_meta, "a") as f:
        f.write(f"MA1,{idx},,{image_meta}/MA1-Figure-Number-Test/{idx}-0.png;{image_meta}/MA1-Figure-Number-Test/{idx}-1.png,{label}\n")

print(">>>>MA1-Figure-Number-Test completed. Starting S1-Card-Rotations-Test ...")

S1_dataset = RandomPolygonDataset(num_polygons=S1_config["num"], eval_num=8, n_vertices_range=(S1_config["min_vertex"], S1_config["max_vertex"]))
for idx, (q_img, imgs, labels) in enumerate(S1_dataset):
    q_img.save(f"{image_meta}/S1-Card-Rotations-Test/{idx}-0.png")
    for q_idx, (img, label) in enumerate(zip(imgs, labels)):
        img.save(f"{image_meta}/S1-Card-Rotations-Test/{idx}-{q_idx+1}.png")
        with open(data_meta, "a") as f:
            f.write(f"S1,{idx},,{image_meta}/S1-Card-Rotations-Test/{idx}-0.png;{image_meta}/S1-Card-Rotations-Test/{idx}-{q_idx+1}.png,{'T' if label else 'F'}\n")

print(">>>>S1-Card-Rotations-Test completed. Starting S2-Cube-Comparisons-Test ...")

record_answer = []
for idx in range(S2_config["num"]):
    cube1, cube2 = generate_cube_pairs()
    same = is_same_cube(cube1, cube2)
    if idx < int(S2_config["num"] / 2):
        while not same:
            cube1, cube2 = generate_cube_pairs()
            same = is_same_cube(cube1, cube2)
    else:
        while same:
            cube1, cube2 = generate_cube_pairs()
            same = is_same_cube(cube1, cube2)
    cv2.imwrite(f"{image_meta}/S2-Cube-Comparisons-Test/{idx}-0.png", generate_cube(cube1))
    cv2.imwrite(f"{image_meta}/S2-Cube-Comparisons-Test/{idx}-1.png", generate_cube(cube2))
    record_answer.append('T' if same else 'F')
    with open(data_meta, "a") as f:
        f.write(f"S2,{idx},,{image_meta}/S2-Cube-Comparisons-Test/{idx}-0.png;{image_meta}/S2-Cube-Comparisons-Test/{idx}-1.png,{'T' if same else 'F'}\n")

for idx, ans in enumerate(record_answer):
    with open(data_meta, "a") as f:
        f.write(f"S2,{idx},,{image_meta}/S2-Cube-Comparisons-Test/{idx}-0.png;{image_meta}/S2-Cube-Comparisons-Test/{idx}-1.png,{'F' if ans else 'T'}\n")

for idx, ans in enumerate(record_answer):
    with open(data_meta, "a") as f:
        f.write(f"S2,{idx},,{image_meta}/S2-Cube-Comparisons-Test/{idx}-0.png;{image_meta}/S2-Cube-Comparisons-Test/{idx}-1.png,{'T' if ans else 'F'}\n")

for idx, ans in enumerate(record_answer):
    with open(data_meta, "a") as f:
        f.write(f"S2,{idx},,{image_meta}/S2-Cube-Comparisons-Test/{idx}-0.png;{image_meta}/S2-Cube-Comparisons-Test/{idx}-1.png,{'F' if ans else 'T'}\n")

print(">>>>S2-Cube-Comparisons-Test completed. Starting SS3-Map-Planning-Test ...")

for idx in range(SS3_config["num"]):
    city = City(
        rows=SS3_config["rows"],
        cols=SS3_config["cols"],
        blocked_ratio=SS3_config["blocked_ratio"],
        num_buildings=SS3_config["num_buildings"]
    )
    while len(city.crossed_buildings) != 1:
        city = City(
            rows=SS3_config["rows"],
            cols=SS3_config["cols"],
            blocked_ratio=SS3_config["blocked_ratio"],
            num_buildings=SS3_config["num_buildings"]
        )
    city.draw(f"{image_meta}/SS3-Map-Planning-Test/{idx}.png")
    with open(data_meta, "a") as f:
        f.write(f"SS3,{idx},{city.start_label} to {city.end_label},{image_meta}/SS3-Map-Planning-Test/{idx}.png,{list(city.crossed_buildings)[0]}\n")
        f.write(f"SS3,{idx},{city.end_label} to {city.start_label},{image_meta}/SS3-Map-Planning-Test/{idx}.png,{list(city.crossed_buildings)[0]}\n")

print(">>>>SS3-Map-Planning-Test completed. Starting VZ1-Form-Board-Test ...")

for idx in range(VZ1_config["num"]):
    answers = Puzzle.from_edges(VZ1_config[f"model_{idx % 4}"]).export(f"{image_meta}/VZ1-Form-Board-Test/{idx}.png")
    for a in answers:
        with open(data_meta, "a") as f:
            f.write(f"VZ1,{idx},,{image_meta}/VZ1-Form-Board-Test/{idx}-0.png;{image_meta}/VZ1-Form-Board-Test/{idx}-choices.png,{a}\n")

print(">>>>VZ1-Form-Board-Test completed. Starting VZ2-Paper-Folding-Test ...")

for idx in range(VZ2_config["num"]):
    generate_sequence(VZ2_config["n"], VZ2_config["min_steps"], VZ2_config["max_steps"], save_dir=f"{image_meta}/VZ2-Paper-Folding-Test/{idx}")
    process_images(f"{image_meta}/VZ2-Paper-Folding-Test/", idx)
    with open(data_meta, "a") as f:
        f.write(f"VZ2,{idx},,{image_meta}/VZ2-Paper-Folding-Test/{idx}_question.png;{image_meta}/VZ2-Paper-Folding-Test/{idx}_correct_choice.png,T\n")
        f.write(f"VZ2,{idx},,{image_meta}/VZ2-Paper-Folding-Test/{idx}_question.png;{image_meta}/VZ2-Paper-Folding-Test/{idx}/wrong_choice_0.png,F\n")
        f.write(f"VZ2,{idx},,{image_meta}/VZ2-Paper-Folding-Test/{idx}_question.png;{image_meta}/VZ2-Paper-Folding-Test/{idx}/wrong_choice_1.png,F\n")
        f.write(f"VZ2,{idx},,{image_meta}/VZ2-Paper-Folding-Test/{idx}_question.png;{image_meta}/VZ2-Paper-Folding-Test/{idx}/wrong_choice_2.png,F\n")
        f.write(f"VZ2,{idx},,{image_meta}/VZ2-Paper-Folding-Test/{idx}_question.png;{image_meta}/VZ2-Paper-Folding-Test/{idx}/wrong_choice_3.png,F\n")

print(">>>>VZ2-Paper-Folding-Test completed. All finished.")
