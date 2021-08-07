import os
import random
import tempfile
import traceback
import dataclasses
from urllib.request import urlopen

import cv2 as cv
import numpy as np
import streamlit as st

from modules import *
from strings import *


# [start] [defaults] ________________________________________________
# local
dataPath = r"H:\0ut51d3r5.17711\_hlu.Projects.Data\open-cv-murtaza-resources"
demoImages = ["reshot01.jpg", "reshot02.jpg", "reshot03.jpg", "reshot04.jpg"]
demoVideos = ["pexels01.mp4", "pexels02.mp4", "pexels04.mp4", "pexels08.mp4"]
demoWebCam = "webcam_image_0.png"
# online
urlImages = [
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838655/photosp/0a0f136f-9032-4480-a1ca-1185dd161368/0a0f136f-9032-4480-a1ca-1185dd161368.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838912/photosp/57ece171-00ea-439c-95e2-01523fd41285/57ece171-00ea-439c-95e2-01523fd41285.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838985/photosp/fc11c636-e5db-4b02-b3d2-99650128c351/fc11c636-e5db-4b02-b3d2-99650128c351.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838983/photosp/22f4d64b-6358-47b5-8a96-a5b9b8019829/22f4d64b-6358-47b5-8a96-a5b9b8019829.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1613176985/photosp/373e53b2-49a8-4e5f-85d2-bfc1a233572e/373e53b2-49a8-4e5f-85d2-bfc1a233572e.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1588713749/photosp/148fb738-110b-45d5-96e5-89bd36335b91/148fb738-110b-45d5-96e5-89bd36335b91.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1541429083/photosp/76400249-2c7c-4030-b381-95c1c4106db6/76400249-2c7c-4030-b381-95c1c4106db6.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1525791591/photosp/ee9ec8f6-3f11-48a6-a112-57825f983b3a/ee9ec8f6-3f11-48a6-a112-57825f983b3a.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838947/photosp/705bc78c-6e43-47c1-8295-802b48106695/705bc78c-6e43-47c1-8295-802b48106695.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1588713965/photosp/4c7d6a68-a215-47bd-a1a6-7fb137cdf6c4/4c7d6a68-a215-47bd-a1a6-7fb137cdf6c4.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1611254271/photosp/c67f2181-50a6-4b4d-9099-d401197a99a2/c67f2181-50a6-4b4d-9099-d401197a99a2.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1623132298/photosp/fafc9d9c-f2dc-46f5-9fa4-885914b176b0/fafc9d9c-f2dc-46f5-9fa4-885914b176b0.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838971/photosp/a62f2758-ca17-424f-b406-d646162202d7/a62f2758-ca17-424f-b406-d646162202d7.jpg",
]
urlVideos = [
    "https://assets.mixkit.co/videos/preview/mixkit-joyful-lovers-hugging-4643-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-friends-with-colored-smoke-bombs-4556-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-group-of-friends-partying-happily-4640-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-silhouette-of-a-couple-jumping-in-the-sunset-33417-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-girl-dancing-happy-at-home-8752-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-multiracial-people-joining-hands-23012-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-a-woman-and-a-man-in-a-beautiful-park-4882-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-mother-with-her-happy-daughters-4549-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-excited-girl-talking-on-video-call-with-her-cell-phone-8745-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-excited-young-people-partying-with-party-hats-4608-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-young-woman-talking-on-video-call-on-a-terrace-10442-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-happy-smiling-students-promote-college-education-9001-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-woman-doing-mountain-climber-exercise-726-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-man-training-on-the-bars-in-the-gym-23450-large.mp4",
]
# global parameters
target_h, target_w = 350, 550

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

# [start] [persistent states]__________________________________________
@dataclasses.dataclass
class functionsState:
    current_image_path: str = demoImages[0]
    current_image_url: str = urlImages[0]
    idx_url_image: int = 0
    current_video_path: str = demoVideos[0]
    current_video_url: str = urlVideos[0]
    idx_url_video: int = 0
    sol_confidence: float = 0.65
    num_hands: int = 2
    smooth_lms: int = 1
    face_model: int = 0
    num_faces: int = 2
    current_image_upload: str = ""
    current_video_upload: str = ""
    uploader_key: int = 0


@st.cache(allow_output_mutation=True)
def _functionsState() -> functionsState:
    return functionsState()


_fs = _functionsState()

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]


def open_img_path_url(url_or_file, source_type, source_path=None, resize=False):
    img, mask = [], []

    if source_type == "path":
        if source_path is None:
            source_path = dataPath
        img = cv.imread(os.path.join(source_path, url_or_file))

    elif source_type == "url":
        resp = urlopen(url_or_file)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv.imdecode(img, cv.IMREAD_COLOR)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    if not resize:
        return img

    else:
        img_h = img.shape[0]
        ratio = target_h / img_h
        r_img = cv.resize(img, None, fx=ratio, fy=ratio)

        r_img_w = r_img.shape[1]
        left_edge = target_w // 2 - r_img_w // 2

        mask = np.zeros((target_h, target_w, 3), dtype="uint8")
        mask[:, left_edge : left_edge + r_img_w] = r_img

        return img, mask


def open_vid_path_url(url_or_file, source_type, source_path=None, preview=False):
    vid, vid_preview = [], []

    if source_type == "path":
        if source_path is None:
            source_path = dataPath
        vid_preview = os.path.join(source_path, url_or_file)

    elif source_type == "url":
        vid_preview = url_or_file

    vid = cv.VideoCapture(vid_preview)

    if preview:
        return vid, vid_preview
    else:
        return vid


def open_webcam():
    vid = cv.VideoCapture(0)
    vid.set(3, 1280)  # width
    vid.set(4, 720)  # height

    return vid


def read_source_media(data_source_selection):
    if data_source_selection == "User Image":
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        img_file_buffer = st.sidebar.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png"], key=_fs.uploader_key
        )

        if img_file_buffer:
            temp_file.write(img_file_buffer.read())
            _fs.current_image_upload = temp_file.name

        if _fs.current_image_upload != "":
            img, mask = open_img_path_url(
                _fs.current_image_upload, "path", source_path="", resize=True
            )

            st.sidebar.markdown("")
            cols = st.sidebar.beta_columns([2, 1])
            cols[0].text("Original Image")
            st.sidebar.image(mask, use_column_width=True)
            if cols[1].button("Clear Upload"):
                _fs.current_image_upload = ""
                _fs.uploader_key += 1
                st.experimental_rerun()
            st.sidebar.markdown("---")

            return img, "image"

    elif data_source_selection == "Random Image Path":  # pass
        img = open_img_path_url(_fs.current_image_path, "path")

        st.sidebar.markdown("")
        cols = st.sidebar.beta_columns([2, 1])
        cols[0].text("Original Image")
        st.sidebar.image(img, use_column_width=True)
        if cols[1].button("Change Image"):
            _fs.current_image_path = random.choice(demoImages)
            st.experimental_rerun()
        st.sidebar.markdown("---")

        return img, "image"

    elif data_source_selection == "Random Image":
        img = open_img_path_url(_fs.current_image_url, "url")

        st.sidebar.markdown("")
        cols = st.sidebar.beta_columns([2, 1])
        cols[0].text("Original Image")
        st.sidebar.image(img, use_column_width=True)
        if cols[1].button("Change Image"):
            _fs.current_image_url = urlImages[_fs.idx_url_image % len(urlImages)]
            _fs.idx_url_image += 1
            st.experimental_rerun()
        st.sidebar.markdown("---")

        return img, "image"

    elif data_source_selection == "User Video":
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        vid_file_buffer = st.sidebar.file_uploader(
            "Upload Video", type=["mp4", "mov", "avi", "3gp", "m4v"], key=_fs.uploader_key
        )

        if vid_file_buffer:
            temp_file.write(vid_file_buffer.read())
            _fs.current_video_upload = temp_file.name

        if _fs.current_video_upload != "":
            vid = open_vid_path_url(_fs.current_video_upload, "path")

            st.sidebar.markdown("")
            cols = st.sidebar.beta_columns([2, 1])
            cols[0].text("Original Video")
            st.sidebar.video(_fs.current_video_upload)
            if cols[1].button("Clear Upload"):
                _fs.current_video_upload = ""
                _fs.uploader_key += 1
                st.experimental_rerun()
            st.sidebar.markdown("---")

            return vid, "video"

    elif data_source_selection == "Random Video Path":  # pass
        vid, vid_preview = open_vid_path_url(_fs.current_video_path, "path", preview=True)

        st.sidebar.markdown("")
        cols = st.sidebar.beta_columns([2, 1])
        cols[0].text("Original Video")
        st.sidebar.video(vid_preview)
        if cols[1].button("Change Video"):
            _fs.current_video_path = random.choice(demoVideos)
            st.experimental_rerun()
        st.sidebar.markdown("---")

        return vid, "video"

    elif data_source_selection == "Random Video":
        vid, vid_preview = open_vid_path_url(_fs.current_video_url, "url", preview=True)

        st.sidebar.markdown("")
        cols = st.sidebar.beta_columns([2, 1])
        cols[0].text("Original Video")
        st.sidebar.video(vid_preview)
        if cols[1].button("Change Video"):
            _fs.current_video_url = urlVideos[_fs.idx_url_video % len(urlVideos)]
            _fs.idx_url_video += 1
            st.experimental_rerun()
        st.sidebar.markdown("---")

        return vid, "video"

    elif data_source_selection == "WebCam":
        vid = open_webcam()

        st.sidebar.text("WebCam")
        st.sidebar.image(open_img_path_url(demoWebCam, "path"))
        st.sidebar.markdown("---")

        return vid, "webcam"

    return None, None  # final fallback


def init_module(media, type, detector, placeholders):
    frame_count = 0
    cols = placeholders[0].beta_columns([2, 2, 1, 1])

    if type == "image":
        img = detector.findFeatures(media)
        placeholders[1].image(img, use_column_width=True)

    if type in ["video", "webcam"]:
        stop_clicked = cols[0].button("🟥 STOP")
        start_clicked = cols[3].button("🟢 START")

        while True:
            try:
                success, img = media.read()
                frame_count += 1

                if type == "webcam":
                    img = cv.flip(img, 1)

                if frame_count == media.get(cv.CAP_PROP_FRAME_COUNT):
                    frame_count = 0
                    media.set(cv.CAP_PROP_POS_FRAMES, 0)

                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = detector.findFeatures(img)

                placeholders[1].image(img, use_column_width=True)

                if stop_clicked:
                    placeholders[1].empty()
                    media.release()
                    cv.destroyAllWindows()
                    break

                if start_clicked:
                    st.experimental_rerun()

            except Exception:
                placeholders[1].info(traceback.format_exc())
                media.release()
                cv.destroyAllWindows()
                break


def run_selected_module(module_selection, media, type, ph_variables):
    moreInfo1 = st.empty()
    moreInfo2 = st.empty()
    moduleOutput1 = st.empty()
    moduleOutput2 = st.empty()

    _fs.sol_confidence = ph_variables[0].slider(
        "Solution Confidence [0.4-1.0]",
        min_value=0.4,
        max_value=1.0,
        value=_fs.sol_confidence,
    )

    if module_selection == "Hand Tracking":
        moreInfo1.markdown(
            "*Click below for information on the Mediapipe **Hands** solution...*"
        )
        new_value = ph_variables[1].number_input(
            "Number Of Hands [1-6]", min_value=1, max_value=6, value=_fs.num_hands
        )
        if new_value != _fs.num_hands:
            _fs.num_hands = new_value
            st.experimental_rerun()

        with moreInfo2.beta_expander(""):
            st.markdown(aboutMpHands(), unsafe_allow_html=True)
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

        detector = handDetector(
            numHands=_fs.num_hands, solutionConfidence=_fs.sol_confidence
        )
        init_module(media, type, detector, (moduleOutput1, moduleOutput2))
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

    elif module_selection == "Pose Estimation":
        moreInfo1.markdown(
            "*Click below for information on the Mediapipe **Pose** solution...*"
        )
        new_value = ph_variables[1].number_input(
            "Smooth Landmarks [0/1]", min_value=0, max_value=1, value=_fs.smooth_lms
        )
        if new_value != _fs.smooth_lms:
            _fs.smooth_lms = new_value
            st.experimental_rerun()

        with moreInfo2.beta_expander(""):
            st.markdown(aboutMpPose(), unsafe_allow_html=True)
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

        detector = poseDetector(
            smoothLandmarks=bool(_fs.smooth_lms), solutionConfidence=_fs.sol_confidence
        )
        init_module(media, type, detector, (moduleOutput1, moduleOutput2))
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

    elif module_selection == "Face Detection":
        moreInfo1.markdown(
            "*Click below for information on the Mediapipe **Face Detection** solution...*"
        )
        new_value = ph_variables[1].number_input(
            "Model Selection [0/1]", min_value=0, max_value=1, value=_fs.face_model
        )
        if new_value != _fs.face_model:
            _fs.face_model = new_value
            st.experimental_rerun()

        with moreInfo2.beta_expander(""):
            st.markdown(aboutMpFaceDetection(), unsafe_allow_html=True)
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

        detector = faceDetector(
            modelSelection=_fs.face_model, solutionConfidence=_fs.sol_confidence
        )
        init_module(media, type, detector, (moduleOutput1, moduleOutput2))
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

    elif module_selection == "Face Mesh":
        moreInfo1.markdown(
            "*Click below for information on the Mediapipe **Face Mesh** solution...*"
        )
        new_value = ph_variables[1].number_input(
            "Number Of Faces [1-5]", min_value=1, max_value=5, value=_fs.num_faces
        )
        if new_value != _fs.num_faces:
            _fs.num_faces = new_value
            st.experimental_rerun()

        with moreInfo2.beta_expander(""):
            st.markdown(aboutMpFaceMesh(), unsafe_allow_html=True)
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

        detector = faceMeshDetector(
            numFaces=_fs.num_faces, solutionConfidence=_fs.sol_confidence
        )
        init_module(media, type, detector, (moduleOutput1, moduleOutput2))
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]
