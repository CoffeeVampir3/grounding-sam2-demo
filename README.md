## How 3:

`pip install -r requirements.txt`
`python download_models.py`
`python interface.py`

The text box controls the grounding, and is what gives you the mask. Type something in and hit ground image. The 2nd from the right image can place control points, you can switch the control points be positive (include this part of the image) or negative (exclude this part of the image.) Control points are optional but will give SAM2 a better chance of segmenting well if you pick good points.

- Grounds things with <https://github.com/IDEA-Research/GroundingDINO>
- Segments things with <https://github.com/facebookresearch/segment-anything-2>
![image](https://github.com/user-attachments/assets/848c0895-a1e9-466a-ab9c-a7411f8e973c)
![image](https://github.com/user-attachments/assets/5a1dc0d9-9cd6-45ae-9124-dd412381e19a)
![image](https://github.com/user-attachments/assets/900daf09-b435-474e-8cac-356d4452218e)
