# Anomaly detection Autoencoder with MSE loss

This is an implementation of L2 Autoencoder for Anomaly detection based on MVTEC-AD dataset.<br>

## How to Train:

use train.py to train the model for a specific object class. Training is customizable
with options for batch_size, epochs, filters, etc from the command-line arguments

### Example:

- **bottle** object
```bash
python train.py --name bottle --epochs 50 --aug_to 10000 --filters 32 64 126 256
```

## How to start fastAPI server:

use server.py to boot up the fastAPI server to infer the model

### Example:

```bash
fastapi dev server.py
```

## Test the API

Use swagger docs for executing image endpoint (127.0.0.1:8000/docs - "for dev mode").
Run api_test.py to check the array endpoint using requests module.

## Notes:

Usually for anomaly detection you might expect one model for all the object class,
while this might work when the requirement is to classify image as anomalous, 
it's not suitable when you have to detect anomalous pixels/areas in the image.

Hence, I've adopted a single-model per object class method, essentially training
an autoencoder for every object-class. Used "MSE" loss.
I've trained the model with limited augmented dataset (2000 total images)
with 256*256 dimensions as this was the maximum I could fit in my memory.
I've trained models for 3 object classes for the prototype.
Training is easily customizable with command-line arguments.
API supports endpoints for image inputs as well as array inputs.

Also for train.py to work the working-directory should contain "data" folder
containing the MVTEC-AD dataset with the following structure

```bash
├───data
│   ├───bottle
│   │   ├───ground_truth
│   │   │   ├───broken_large
│   │   │   ├───broken_small
│   │   │   └───contamination
│   │   ├───test
│   │   │   ├───broken_large
│   │   │   ├───broken_small
│   │   │   ├───contamination
│   │   │   └───good
│   │   └───train
│   │       └───good
│   ├───cable
│   │   ├───ground_truth
│   │   │   ├───bent_wire
│   │   │   ├───cable_swap
│   │   │   ├───combined
│   │   │   ├───cut_inner_insulation
│   │   │   ├───cut_outer_insulation
│   │   │   ├───missing_cable
│   │   │   ├───missing_wire
│   │   │   └───poke_insulation
│   │   ├───test
│   │   │   ├───bent_wire
│   │   │   ├───cable_swap
│   │   │   ├───combined
│   │   │   ├───cut_inner_insulation
│   │   │   ├───cut_outer_insulation
│   │   │   ├───good
│   │   │   ├───missing_cable
│   │   │   ├───missing_wire
│   │   │   └───poke_insulation
│   │   └───train
│   │       └───good
│   ├───capsule
│   │   ├───ground_truth
│   │   │   ├───crack
│   │   │   ├───faulty_imprint
│   │   │   ├───poke
│   │   │   ├───scratch
│   │   │   └───squeeze
│   │   ├───test
│   │   │   ├───crack
│   │   │   ├───faulty_imprint
│   │   │   ├───good
│   │   │   ├───poke
│   │   │   ├───scratch
│   │   │   └───squeeze
│   │   └───train
│   │       └───good
│   ├───carpet
│   │   ├───ground_truth
│   │   │   ├───color
│   │   │   ├───cut
│   │   │   ├───hole
│   │   │   ├───metal_contamination
│   │   │   └───thread
│   │   ├───test
│   │   │   ├───color
│   │   │   ├───cut
│   │   │   ├───good
│   │   │   ├───hole
│   │   │   ├───metal_contamination
│   │   │   └───thread
│   │   └───train
│   │       └───good
│   ├───grid
│   │   ├───ground_truth
│   │   │   ├───bent
│   │   │   ├───broken
│   │   │   ├───glue
│   │   │   ├───metal_contamination
│   │   │   └───thread
│   │   ├───test
│   │   │   ├───bent
│   │   │   ├───broken
│   │   │   ├───glue
│   │   │   ├───good
│   │   │   ├───metal_contamination
│   │   │   └───thread
│   │   └───train
│   │       └───good
│   ├───hazelnut
│   │   ├───ground_truth
│   │   │   ├───crack
│   │   │   ├───cut
│   │   │   ├───hole
│   │   │   └───print
│   │   ├───test
│   │   │   ├───crack
│   │   │   ├───cut
│   │   │   ├───good
│   │   │   ├───hole
│   │   │   └───print
│   │   └───train
│   │       └───good
│   ├───leather
│   │   ├───ground_truth
│   │   │   ├───color
│   │   │   ├───cut
│   │   │   ├───fold
│   │   │   ├───glue
│   │   │   └───poke
│   │   ├───test
│   │   │   ├───color
│   │   │   ├───cut
│   │   │   ├───fold
│   │   │   ├───glue
│   │   │   ├───good
│   │   │   └───poke
│   │   └───train
│   │       └───good
│   ├───metal_nut
│   │   ├───ground_truth
│   │   │   ├───bent
│   │   │   ├───color
│   │   │   ├───flip
│   │   │   └───scratch
│   │   ├───test
│   │   │   ├───bent
│   │   │   ├───color
│   │   │   ├───flip
│   │   │   ├───good
│   │   │   └───scratch
│   │   └───train
│   │       └───good
│   ├───pill
│   │   ├───ground_truth
│   │   │   ├───color
│   │   │   ├───combined
│   │   │   ├───contamination
│   │   │   ├───crack
│   │   │   ├───faulty_imprint
│   │   │   ├───pill_type
│   │   │   └───scratch
│   │   ├───test
│   │   │   ├───color
│   │   │   ├───combined
│   │   │   ├───contamination
│   │   │   ├───crack
│   │   │   ├───faulty_imprint
│   │   │   ├───good
│   │   │   ├───pill_type
│   │   │   └───scratch
│   │   └───train
│   │       └───good
│   ├───screw
│   │   ├───ground_truth
│   │   │   ├───manipulated_front
│   │   │   ├───scratch_head
│   │   │   ├───scratch_neck
│   │   │   ├───thread_side
│   │   │   └───thread_top
│   │   ├───test
│   │   │   ├───good
│   │   │   ├───manipulated_front
│   │   │   ├───scratch_head
│   │   │   ├───scratch_neck
│   │   │   ├───thread_side
│   │   │   └───thread_top
│   │   └───train
│   │       └───good
│   ├───tile
│   │   ├───ground_truth
│   │   │   ├───crack
│   │   │   ├───glue_strip
│   │   │   ├───gray_stroke
│   │   │   ├───oil
│   │   │   └───rough
│   │   ├───test
│   │   │   ├───crack
│   │   │   ├───glue_strip
│   │   │   ├───good
│   │   │   ├───gray_stroke
│   │   │   ├───oil
│   │   │   └───rough
│   │   └───train
│   │       └───good
│   ├───toothbrush
│   │   ├───ground_truth
│   │   │   └───defective
│   │   ├───test
│   │   │   ├───defective
│   │   │   └───good
│   │   └───train
│   │       └───good
│   ├───transistor
│   │   ├───ground_truth
│   │   │   ├───bent_lead
│   │   │   ├───cut_lead
│   │   │   ├───damaged_case
│   │   │   └───misplaced
│   │   ├───test
│   │   │   ├───bent_lead
│   │   │   ├───cut_lead
│   │   │   ├───damaged_case
│   │   │   ├───good
│   │   │   └───misplaced
│   │   └───train
│   │       └───good
│   ├───wood
│   │   ├───ground_truth
│   │   │   ├───color
│   │   │   ├───combined
│   │   │   ├───hole
│   │   │   ├───liquid
│   │   │   └───scratch
│   │   ├───test
│   │   │   ├───color
│   │   │   ├───combined
│   │   │   ├───good
│   │   │   ├───hole
│   │   │   ├───liquid
│   │   │   └───scratch
│   │   └───train
│   │       └───good
│   └───zipper
│       ├───ground_truth
│       │   ├───broken_teeth
│       │   ├───combined
│       │   ├───fabric_border
│       │   ├───fabric_interior
│       │   ├───rough
│       │   ├───split_teeth
│       │   └───squeezed_teeth
│       ├───test
│       │   ├───broken_teeth
│       │   ├───combined
│       │   ├───fabric_border
│       │   ├───fabric_interior
│       │   ├───good
│       │   ├───rough
│       │   ├───split_teeth
│       │   └───squeezed_teeth
│       └───train
│           └───good
```
