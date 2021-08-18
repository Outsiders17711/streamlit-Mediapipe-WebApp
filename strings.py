import streamlit as st


pageConfig = """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """

nbsp = "&nbsp"

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]


@st.cache(allow_output_mutation=True, max_entries=3, ttl=3600)
def aboutMe():
    return """
        ## **Hi 👋 I'm Umar...**

        <div style="text-align: justify;">

        I'm a highly resourceful mechanical engineer with over three years of industry experience in construction and telecommunications. I'm currently doing my masters in Robotics, Control and Smart Systems at the American University in Cairo.

        You can check out my personal [![](https://img.shields.io/static/v1?label=GitHub%20Pages&message=Blog&labelColor=2f363d&color=blue&style=flat&logo=github)](https://outsiders17711.github.io/Mein.Platz/) where I detail my experiences and development with `Machine Learning`, `Computer Vision` and `Gesture Recognition` as I work on my masters.

        </div>

        <hr>
        
        <div style="text-align: center;">

        ## 🔭 **Tools of Trade**

        🛠 **Programming**
        
        ![](https://img.shields.io/badge/python-%2314354C.svg?style=flat&logo=python&logoColor=white) | ![](https://img.shields.io/badge/markdown-%23000000.svg?style=flat&logo=markdown&logoColor=white) | ![](https://img.shields.io/badge/VisualStudioCode-0078d7.svg?style=flat&logo=visual-studio-code&logoColor=white) | ![](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white) | ![](https://img.shields.io/badge/git-%23F05033.svg?style=flat&logo=git&logoColor=white) | ![](https://img.shields.io/badge/github-%23121011.svg?style=flat&logo=github&logoColor=white)
        
        <br>

        🛠 **Machine Learning & Computer Vision**
        
        ![](https://img.shields.io/badge/Keras-%23D00000.svg?style=flat&logo=Keras&logoColor=white) | ![](https://img.shields.io/badge/opencv-%23white.svg?style=flat&logo=opencv&logoColor=white) | ![](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white) | ![](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) | ![](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white) | ![](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)

        </div>

        <hr>
        """


@st.cache(allow_output_mutation=True, max_entries=3, ttl=3600)
def aboutWebApp():
    return (
        """
    This application explores the functionality of some of Google's <a href="https://google.github.io/mediapipe/" style="text-decoration: none;">**Mediapipe**</a> Machine Learning solutions, viz:

    - Hand Tracking
    - Pose Estimation
    - Face Detection 
    - Face Mesh
    
    <a href="https://streamlit.io/" style="text-decoration: none;">**StreamLit**</a> is used to create the Web Graphical User Interface (GUI). Streamlit is a cool way to turn data scripts into shareable web apps in minutes, all in Python. You can check out the launch <a href="https://towardsdatascience.com/coding-ml-tools-like-you-code-ml-models-ddba3357eace" style="text-decoration: none;">*blog post*</a> for more information.

    This web app was inspired by the awesome YouTube tutorials created by:
    
    - **Murtaza Hassan**: <a href="https://www.youtube.com/watch?v=01sAkU_NvOY" style="text-decoration: none;">Advanced Computer Vision with Python - Full Course</a>
    - **Augmented Startups**: <a href="https://www.youtube.com/watch?v=wyWmWaXapmI" style="text-decoration: none;">StreamLit Computer Vision User Interface Course | MediaPipe OpenCV Python (2021)</a>

    """,
        """

    Do check out their channels and websites for more informative and exciting Machine Learning & Computer Vision tutorials.

    <hr>
    """,
    )


@st.cache(allow_output_mutation=True, max_entries=3, ttl=3600)
def aboutMpHands():
    return """
    <style>
    img.centered {
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-top: 10px;
        margin-bottom: 2.5px;
    }

    p.caption {
        font-style: italic;
        font-size: smaller;
        font-variant: small-caps;
    }
    </style>

    <div style="text-align: justify;">
    <a href="https://google.github.io/mediapipe/solutions/hands.html" style="text-decoration: none;"><b>MediaPipe Hands</b></a> is a high-fidelity hand and finger tracking solution. It employs machine learning (ML) to infer 21 3D landmarks of a hand from just a single frame.
    </div>

    <div style="text-align: center;">
    <img src="https://google.github.io/mediapipe/images/mobile/hand_landmarks.png" class="centered" width=650px>
    <p class="caption" >21 hand landmarks.</p>
    </div>
    
    <div style="text-align: justify;">
    MediaPipe Hands utilizes an ML pipeline consisting of multiple models working together: A palm detection model that operates on the full image and returns an oriented hand bounding box. A hand landmark model that operates on the cropped image region defined by the palm detector and returns high-fidelity 3D hand keypoints. 
    </div>

    <div style="text-align: center;">
    <img src="https://google.github.io/mediapipe/images/mobile/hand_tracking_3d_android_gpu.gif" class="centered">
    <p class="caption" >Tracked 3D hand landmarks are represented by dots in different shades, with the brighter ones denoting landmarks closer to the camera.</p>
    </div>
    
    """


@st.cache(allow_output_mutation=True, max_entries=3, ttl=3600)
def aboutMpPose():
    return """
    <style>
    img.centered {
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-top: 10px;
        margin-bottom: 2.5px;
    }

    p.caption {
        font-style: italic;
        font-size: smaller;
        font-variant: small-caps;
    }
    </style>

    <div style="text-align: justify;">
    <a href="https://google.github.io/mediapipe/solutions/pose.html" style="text-decoration: none;"><b>MediaPipe Pose</b></a> is a ML solution for high-fidelity body pose tracking, inferring 33 3D landmarks on the whole body from RGB video frames.
    </div>

    <div style="text-align: center;">
    <img src="https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png" class="centered" width=650px>
    <p class="caption" >33 pose landmarks.</p>
    </div>
    
    <div style="text-align: justify;">
    The solution utilizes a two-step detector-tracker ML pipeline. Using a detector, the pipeline first locates the person/pose region-of-interest (ROI) within the frame. The tracker subsequently predicts the pose landmarks within the ROI using the ROI-cropped frame as input. Note that for video use cases the detector is invoked only as needed, i.e., for the very first frame and when the tracker could no longer identify body pose presence in the previous frame. For other frames the pipeline simply derives the ROI from the previous frame’s pose landmarks.
    </div>

    <div style="text-align: center;">
    <img src="https://google.github.io/mediapipe/images/mobile/pose_tracking_example.gif" class="centered">
    <p class="caption" >Example of MediaPipe Pose for pose tracking.</p>
    </div>
    
    """


@st.cache(allow_output_mutation=True, max_entries=3, ttl=3600)
def aboutMpFaceDetection():
    return """
    <style>
    img.centered {
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-top: 10px;
        margin-bottom: 2.5px;
    }

    p.caption {
        font-style: italic;
        font-size: smaller;
        font-variant: small-caps;
    }
    </style>

    <div style="text-align: justify;">
    <a href="https://google.github.io/mediapipe/solutions/face_detection.html" style="text-decoration: none;"><b>MediaPipe Face Detection</b></a> is an ultrafast face detection solution that comes with 6 landmarks and multi-face support. The detector’s super-realtime performance enables it to be applied to any live viewfinder experience that requires an accurate facial region of interest as an input for other task-specific models.<br><br>
    The solution provides two model types: <code>model 0</code> to select a short-range model that works best for faces within 2 meters from the camera, and <code>model 1</code> for a full-range model best for faces within 5 meters.<br><br>
    The output is a collection of detected faces, where each face is represented as a detection proto message that contains a bounding box and 6 key points (right eye, left eye, nose tip, mouth center, right ear tragion, and left ear tragion). 
    </div>

    <div style="text-align: center;">
    <img src="https://google.github.io/mediapipe/images/mobile/face_detection_android_gpu.gif" class="centered">
    <p class="caption" > </p>
    </div>
    
    """


@st.cache(allow_output_mutation=True, max_entries=3, ttl=3600)
def aboutMpFaceMesh():
    return """
    <style>
    img.centered {
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-top: 10px;
        margin-bottom: 2.5px;
    }

    p.caption {
        font-style: italic;
        font-size: smaller;
        font-variant: small-caps;
    }
    </style>

    <div style="text-align: justify;">
    <a href="https://google.github.io/mediapipe/solutions/face_mesh.html" style="text-decoration: none;"><b>MediaPipe Face Mesh</b></a> is a face geometry solution that estimates 468 3D face landmarks in real-time even on mobile devices. It employs machine learning (ML) to infer the 3D surface geometry, requiring only a single camera input without the need for a dedicated depth sensor.<br><br>
    The ML pipeline consists of two real-time deep neural network models that work together: A detector that operates on the full image and computes face locations and a 3D face landmark model that operates on those locations and predicts the approximate surface geometry via regression. Having the face accurately cropped drastically reduces the need for common data augmentations like affine transformations consisting of rotations, translation and scale changes. Instead it allows the network to dedicate most of its capacity towards coordinate prediction accuracy.
    </div>

    <div style="text-align: center;">
    <img src="https://google.github.io/mediapipe/images/mobile/face_mesh_android_gpu.gif" class="centered">
    <p class="caption" >Face landmarks: the red box indicates the cropped area as input to the landmark model, the red dots represent the 468 landmarks in 3D, and the green lines connecting landmarks illustrate the contours around the eyes, eyebrows, lips and the entire face.</p>
    </div>
    
    """


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]
