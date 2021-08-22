import os
import random
import tempfile
import traceback
from urllib.request import urlopen
import shutil

import cv2 as cv
import numpy as np
import streamlit as st

from modules import *
from strings import *


import gc  # garbage collection

gc.enable()


# [start] [defaults] ________________________________________________
# local
dataPath = r"data"
demoImages = ["reshot01.jpg", "reshot02.jpg", "reshot03.jpg", "reshot04.jpg"]
demoVideos = ["pexels03.mp4", "pexels04.mp4", "pexels05.mp4", "pexels08.mp4"]
demoWebCam = "webcam_image.png"
# online
urlImages = [
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1623132298/photosp/fafc9d9c-f2dc-46f5-9fa4-885914b176b0/fafc9d9c-f2dc-46f5-9fa4-885914b176b0.jpg",
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
# webapp
appPages = ["Home Page", "Mediapipe Modules", "About Me"]
appModules = ["Hand Tracking", "Pose Estimation", "Face Detection", "Face Mesh"]

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

        try:
            r_img_w = r_img.shape[1]
            left_edge = target_w // 2 - r_img_w // 2

            mask = np.zeros((target_h, target_w, 3), dtype="uint8")
            mask[:, left_edge : left_edge + r_img_w] = r_img

            return img, mask

        except Exception:
            return img, r_img


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


def open_webcam(device_id):
    vid = cv.VideoCapture(device_id)
    vid.set(3, 1280)  # width
    vid.set(4, 720)  # height

    return vid


def init_module(media, type, detector, placeholders):
    cols = placeholders[0].columns([2, 2, 1, 1])

    if type == "image":
        img = detector.findFeatures(media)
        placeholders[1].columns([2, 10, 2])[1].image(img, use_column_width=True)

        del img  # garbage collection

    if type in ["video", "webcam"]:
        stop_clicked = cols[3].button("🟥 STOP")
        start_clicked = cols[0].button("🟢 START")

        codec = cv.VideoWriter_fourcc(*"avc1")  # *"mp4v" doesn't play with streamlit
        vid_w = int(media.get(cv.CAP_PROP_FRAME_WIDTH))
        vid_h = int(media.get(cv.CAP_PROP_FRAME_HEIGHT))
        vid_fps = int(media.get(cv.CAP_PROP_FPS))
        max_frames = 10 * vid_fps
        frame_count, out_vid_frame_count = 0, 0

        temp_dir = os.path.join(os.path.dirname(__file__), "output_vid")
        if not os.path.lexists(temp_dir):
            os.mkdir(temp_dir)
        out_vid_file = os.path.join(temp_dir, "vid_output.mp4")
        out_vid = cv.VideoWriter(out_vid_file, codec, vid_fps, (vid_w, vid_h))
        # st.write(temp_dir, out_vid_file)

        placeholders[1].info(
            "Click **START** to process video input. Output video length is capped at 15 seconds. You can use the **STOP** button to cancel processing."
        )
        if start_clicked:
            placeholders[1].info("Processing video input...")
            progress_bar = st.progress(0)
            vid_player = st.columns([2, 10, 2])

            while media.isOpened():
                try:
                    success, img = media.read()
                    frame_count += 1
                    out_vid_frame_count += 1
                    progress_bar.progress(int((out_vid_frame_count / max_frames) * 100))

                    if type == "webcam":
                        img = cv.flip(img, 1)

                    if frame_count == media.get(cv.CAP_PROP_FRAME_COUNT):
                        frame_count = 0
                        media.set(cv.CAP_PROP_POS_FRAMES, 0)

                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = detector.findFeatures(img)

                    # placeholders[1].columns([2, 10, 2])[1].image(img, use_column_width=True)
                    out_vid.write(cv.cvtColor(img, cv.COLOR_RGB2BGR))

                    del img  # garbage collection

                    if stop_clicked or out_vid_frame_count == max_frames:
                        placeholders[1].empty()
                        media.release()
                        out_vid.release()
                        st.success(
                            f"**Input video processed successfully! Use the player controls to play and download the processed video!**"
                        )
                        break

                except Exception:
                    placeholders[1].info(traceback.format_exc())
                    media.release()
                    out_vid.release()
                    shutil.rmtree(temp_dir, ignore_errors=True)  # garbage collection
                    break

            vid_player[1].video(out_vid_file)
            shutil.rmtree(temp_dir, ignore_errors=True)  # garbage collection


def run_selected_module(_fs, media, type, ph_variables):
    moreInfo1 = st.empty()
    moreInfo2 = st.empty()
    moduleOutput1 = st.empty()
    moduleOutput2 = st.empty()

    new_value = ph_variables[0].slider(
        "Solution Confidence [0.4-1.0]",
        min_value=0.4,
        max_value=1.0,
        value=_fs.sol_confidence,
    )
    if new_value != _fs.sol_confidence:
        _fs.sol_confidence = new_value
        st.experimental_rerun()

    module_selection = appModules[_fs.idx_current_module]
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

        with moreInfo2.expander(""):
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

        with moreInfo2.expander(""):
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

        with moreInfo2.expander(""):
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

        with moreInfo2.expander(""):
            st.markdown(aboutMpFaceMesh(), unsafe_allow_html=True)
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

        detector = faceMeshDetector(
            numFaces=_fs.num_faces, solutionConfidence=_fs.sol_confidence
        )
        init_module(media, type, detector, (moduleOutput1, moduleOutput2))
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

    # garbage collection
    del (
        moreInfo1,
        moreInfo2,
        moduleOutput1,
        moduleOutput2,
        detector,
    )


def read_source_media(_fs, appSources, ph_variables):
    if _fs.current_image_path == "":
        _fs.current_image_path = demoImages[0]
        _fs.current_image_url = urlImages[0]
        _fs.current_video_path = demoVideos[0]
        _fs.current_video_url = urlVideos[0]

    data_source_selection = appSources[_fs.idx_data_source]
    if data_source_selection == "User Image":
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        img_file_buffer = st.sidebar.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png"], key=f"img_ul_{_fs.uploader_key}"
        )

        if img_file_buffer:
            temp_file.write(img_file_buffer.read())
            _fs.current_image_upload = temp_file.name

        if _fs.current_image_upload != "":
            img, mask = open_img_path_url(
                _fs.current_image_upload, "path", source_path="", resize=True
            )

            st.sidebar.markdown("")
            cols = st.sidebar.columns([3, 2])
            cols[0].text("Original Image")
            st.sidebar.image(mask, use_column_width=True)
            if cols[1].button("Clear Upload"):
                _fs.current_image_upload = ""
                _fs.uploader_key += 1
                st.experimental_rerun()

            del temp_file, img_file_buffer, mask, cols  # garbage collection

            run_selected_module(_fs, img, "image", ph_variables)

    elif data_source_selection == "Local Image":
        img = open_img_path_url(_fs.current_image_path, "path")

        st.sidebar.markdown("")
        cols = st.sidebar.columns([3, 2])
        cols[0].text("Original Image")
        st.sidebar.image(img, use_column_width=True)
        if cols[1].button("Change Image"):
            _fs.current_image_path = random.choice(demoImages)
            st.experimental_rerun()

        del cols  # garbage collection

        run_selected_module(_fs, img, "image", ph_variables)

    elif data_source_selection == "Online Image":
        img = open_img_path_url(_fs.current_image_url, "url")

        st.sidebar.markdown("")
        cols = st.sidebar.columns([3, 2])
        cols[0].text("Original Image")
        st.sidebar.image(img, use_column_width=True)
        if cols[1].button("Change Image"):
            _fs.idx_url_image += 1
            _fs.current_image_url = urlImages[_fs.idx_url_image % len(urlImages)]
            st.experimental_rerun()

        del cols  # garbage collection

        run_selected_module(_fs, img, "image", ph_variables)

    elif data_source_selection == "User Video":
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        vid_file_buffer = st.sidebar.file_uploader(
            "Upload Video",
            type=["mp4", "mov", "avi", "3gp", "m4v"],
            key=f"vid_ul_{_fs.uploader_key}",
        )

        if vid_file_buffer:
            temp_file.write(vid_file_buffer.read())
            _fs.current_video_upload = temp_file.name

        if _fs.current_video_upload != "":
            vid = open_vid_path_url(_fs.current_video_upload, "path")

            st.sidebar.markdown("")
            cols = st.sidebar.columns([3, 2])
            cols[0].text("Original Video")
            st.sidebar.video(_fs.current_video_upload)
            if cols[1].button("Clear Upload"):
                _fs.current_video_upload = ""
                _fs.uploader_key += 1
                st.experimental_rerun()

            del temp_file, vid_file_buffer, cols  # garbage collection

            run_selected_module(_fs, vid, "video", ph_variables)

    elif data_source_selection == "Local Video":
        vid, vid_preview = open_vid_path_url(_fs.current_video_path, "path", preview=True)

        st.sidebar.markdown("")
        cols = st.sidebar.columns([3, 2])
        cols[0].text("Original Video")
        st.sidebar.video(vid_preview)
        if cols[1].button("Change Video"):
            _fs.current_video_path = random.choice(demoVideos)
            st.experimental_rerun()

        del vid_preview, cols  # garbage collection

        run_selected_module(_fs, vid, "video", ph_variables)

    elif data_source_selection == "Online Video":
        vid, vid_preview = open_vid_path_url(_fs.current_video_url, "url", preview=True)

        st.sidebar.markdown("")
        cols = st.sidebar.columns([3, 2])
        cols[0].text("Original Video")
        st.sidebar.video(vid_preview)
        if cols[1].button("Change Video"):
            _fs.idx_url_video += 1
            _fs.current_video_url = urlVideos[_fs.idx_url_video % len(urlVideos)]
            st.experimental_rerun()

        del vid_preview, cols  # garbage collection

        run_selected_module(_fs, vid, "video", ph_variables)

    elif data_source_selection == "WebCam":
        vid = open_webcam(_fs.webcam_device_id)

        cols = st.sidebar.columns([3, 2])
        cols[0].markdown(f"**Webcam**: _Device ID = {_fs.webcam_device_id}_")
        st.sidebar.image(open_img_path_url(demoWebCam, "path"))
        if cols[1].button(f"Switch Device"):
            _fs.webcam_device_id = 0 if _fs.webcam_device_id == 1 else 1
            st.experimental_rerun()

        del cols  # garbage collection

        run_selected_module(_fs, vid, "webcam", ph_variables)

    elif data_source_selection == "Video Sources":
        st.sidebar.image(
            "https://previews.123rf.com/images/pockygallery/pockygallery1607/pockygallery160700075/63949628-inactive-red-stamp-text-on-white.jpg",
            caption="Source: https://www.123rf.com/photo_63949628_stock-vector-inactive-red-stamp-text-on-white.html",
        )

        st.info(
            f""" 
        ### {nbsp*20}**VIDEO SOURCES DISABLED ON STREAMLIT SHARE**
        ---
        There have been a couple of issues running video sources on the online shared app:

        1. There is a considerable lag when running the Mediapipe modules on videos with one viewer/user. 
        2.  If multiple viewers/users attempt to use video inputs at the same time, the app becomes totally unresponsive and needs to be rebooted.

        I suspect that both issues are related but I have been unable to get them fixed. Kindly reach out on [GitHub](https://github.com/Outsiders17711) or [Streamlit](https://discuss.streamlit.io/t/streamlit-app-crashes-unexpectedly/15994) if you have any ideas/suggestions on how I can go about resolving these issues.
        
        ---

        **If you want to test out video sources, you can clone the [source repository](https://github.com/Outsiders17711/streamlit-Mediapipe-WebApp), install the requirements and run `streamlitMediapipe.py`.**

        ---

        Thanks.

        """
        )

    return _fs


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]
