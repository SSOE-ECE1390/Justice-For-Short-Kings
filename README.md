# Justice for Short Kings - The Automatic Height Equalizer You Never Knew You Needed!

<!-- This sets up to run the tests and place a badge on GitHub if it passes -->


![Tests](https://github.com/SSOE-ECE1390/ExampleTeam/actions/workflows/tests.yml/badge.svg)


Brief Description:
The goal of this project is to bring prosperity to the short kings around the world. With one quick snapshot, we find the shortest person in a photo and equalize their height to match the average height of the group by various user-selected methods (e.g., a stool, a hat, or a height adjustment procedure of the legs or torso).  

Team Members:
 - Connor Murray - cjm308@pitt.edu
 - William (Will) Muckelroy III - wlm14@pitt.edu
 - Iyan Nekib - iyn1@pitt.edu
 - Yoel Tamar - yot22@pitt.edu
 - Raymond Zheng - raz43@pitt.edu

## Major Milestones
* Detection of the Person(s)
* Evaluation of Heights within the Image
* Implementing a GUI that allows the user to select their method of height adjustment
* Implementation of the actual modification of the image
* Bonus: Equalize ALL the heights to the same height via shortening and lengthening

## Height Equalizer (Iyan)
The module under `src/Iyan` detects every visible person in a photo, finds the shortest one, and either stretches their upper body or adds a configurable accessory so they match a reference person's height.

1. Download a MediaPipe pose landmarker model (e.g. `pose_landmarker_full.task` from the [MediaPipe release assets](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)).
2. Run the helper script:

   ```
   python -m src.Iyan.hi path/to/input.jpg path/to/output.jpg --model path/to/pose_landmarker_full.task --method accessory
   ```

   - Use `--method stretch` to scale the shortest person's upper body.
   - Supply `--reference-index` to pick who should be matched (defaults to the tallest).
   - Optionally pass `--accessory path/to/hat.png` to use a custom transparent PNG.

The core logic lives in `src/Iyan/height_equalizer.py` and can be imported elsewhere to plug into a GUI or pipeline.

## File Descriptions
This project contains a number of additional files that are used by GitHub to provide information and do tests on code.

### Markup files (*.md)
Markup files, such as this README file, are shown on the home page of GitHub

[Here is a good reference for how to use markup files](https://github.com/lifeparticle/Markdown-Cheatsheet)

* README.md; This file usually holds information about the purpose of the repo, the authors, etc.  

* CODE_OF_CONDUCT.md; This file establishes a set of behavioral expectations for contributors and community members, promoting a positive and inclusive environment.

* LICENSE.md; This file specifies the licensing terms under which your project is released, informing users about how they can use, modify, and distribute your code.

### .gitignore
The .gitignore file is used to specify any files that should not be included in git commits/pushes.  Generally, these are temporary files or specific to your computer.  In this case, I have all the python environment files in the .venc folder flagged to be ignored.

### requirements.txt
The requirements.txt file is a way to specify the libraries needed by python by your code.  Here I have a general use one "requirements.txt" and one specifically used in the code regression testing "requirements_dev.txt".  Once you have your python install setup and running the way you like it, you can automatically generate the requirements.txt file for others to replicate your setup using the command

```
    pip freeze > requirements.txt
```

To install from a requirements.txt file use
```
    pip install -r requirements.txt
```
