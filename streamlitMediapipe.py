import gc  # garbage collection

import streamlit as st
import streamlit.report_thread as ReportThread
from streamlit.server.server import Server
from streamlit import caching

from functions import *
from appSessionState import getSessionState


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]
# [start] [persistent states]__________________________________________
gc.enable()  # garbage collection

webapp = getSessionState(
    # webapp
    idx_current_page=1,
    idx_current_module=2,
    idx_data_source=1,
    # functions
    current_image_path="",
    current_image_url="",
    idx_url_image=0,
    current_video_path="",
    current_video_url="",
    idx_url_video=0,
    sol_confidence=0.65,
    num_hands=2,
    smooth_lms=1,
    face_model=0,
    num_faces=2,
    current_image_upload="",
    current_video_upload="",
    uploader_key=0,
    webcam_device_id=0,
)


def reload():
    caching.clear_cache()
    gc.collect()  # garbage collection
    # webapp
    webapp.idx_current_page = 1
    webapp.idx_current_module = 2
    webapp.idx_data_source = 1
    # functions
    webapp.current_image_path = ""
    webapp.current_image_url = ""
    webapp.idx_url_image = 0
    webapp.current_video_path = ""
    webapp.current_video_url = ""
    webapp.idx_url_video = 0
    webapp.sol_confidence = 0.65
    webapp.num_hands = 2
    webapp.smooth_lms = 1
    webapp.face_model = 0
    webapp.num_faces = 2
    webapp.current_image_upload = ""
    webapp.current_video_upload = ""
    webapp.uploader_key = 0
    webapp.webcam_device_id = 0
    #
    st.experimental_rerun()


appPages = ["Home Page", "Mediapipe Modules", "About Me"]
appModules = ["Hand Tracking", "Pose Estimation", "Face Detection", "Face Mesh"]
appSources = [
    "User Image",
    "Local Image",
    "Online Image",
    "User Video",
    "Local Video",
    "Online Video",
    "WebCam",
    # "Video Sources",
]
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]


# [start] [setup main page and side bar] ____________________________
st.set_page_config(page_title="Streamlit Mediapipe WebApp", layout="wide")
st.set_option("deprecation.showfileUploaderEncoding", False)

st.markdown(
    f"""
    {pageConfig}
    <div style="text-align:center; margin-top:-75px; ">
    <h1 style="font-variant: small-caps; font-size: xx-large; margin-bottom:-45px;" >
    <font color=#ea0525>w e b {nbsp*2} a p p</font>
    </h1>
    <h1> Streamlit Mediapipe Webapp </h1>
    <hr>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"""
    <div style="text-align:center; margin-top:-75px; margin-bottom:20px; ">
    <h3 style="font-variant: small-caps; font-size: xx-large; ">
    <font color=#ea0525>s i d e {nbsp} b a r</font>
    </h3>
    <code style="font-size:small; ">{ReportThread.get_report_ctx().session_id}</code>
    </div>
    """,
    unsafe_allow_html=True,
)
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

# [start] [setup app pages, modules & data sources]__________________
pages = st.sidebar.columns([1, 1, 1])

if pages[1].button("About Me"):
    webapp.idx_current_page = appPages.index("About Me")
    st.experimental_rerun()

if pages[2].button("Reload App"):
    reload()

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

if webapp.idx_current_page == appPages.index("About Me"):
    if pages[0].button("Home Page"):
        webapp.idx_current_page = appPages.index("Home Page")
        st.experimental_rerun()

    st.markdown(aboutMe(), unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.image(open_img_path_url("highlord.jpg", "path"), use_column_width="auto")
    st.sidebar.markdown("---")

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]
elif webapp.idx_current_page == appPages.index("Home Page"):
    if pages[0].button("Mediapipe"):
        webapp.idx_current_page = appPages.index("Mediapipe Modules")
        st.experimental_rerun()

    if st.sidebar.columns([3, 15, 2])[1].button("📌📌 Mediapipe Modules 📌📌"):
        webapp.idx_current_page = appPages.index("Mediapipe Modules")
        st.experimental_rerun()

    st.markdown(aboutWebApp()[0], unsafe_allow_html=True)

    st.sidebar.image(open_img_path_url("mediapipe.jpg", "path"), use_column_width="auto")
    st.sidebar.markdown(
        """
        - <a href="https://google.github.io/mediapipe/#ml-solutions-in-mediapipe" style="text-decoration:none;">ML solutions in MediaPipe</a>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.image(
        open_img_path_url("mediapipe_solutions.jpg", "path"), use_column_width="auto"
    )

    vid1, vid2 = st.columns([1, 1])
    vid1.video("https://www.youtube.com/watch?v=01sAkU_NvOY")
    vid1.caption("Advanced Computer Vision with Python - Full Course")
    vid2.video("https://www.youtube.com/watch?v=wyWmWaXapmI")
    vid2.caption(
        "StreamLit Computer Vision User Interface Course | MediaPipe OpenCV Python (2021)"
    )

    st.markdown(aboutWebApp()[1], unsafe_allow_html=True)


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]
elif webapp.idx_current_page == appPages.index("Mediapipe Modules"):
    if pages[0].button("Home Page"):
        webapp.idx_current_page = appPages.index("Home Page")
        st.experimental_rerun()

    st.sidebar.write("")
    mp_selectors = st.sidebar.columns([1, 1])

    module_selection = mp_selectors[0].selectbox(
        "Mediapipe Solution:",
        appModules,
        index=webapp.idx_current_module,
    )
    if module_selection != appModules[webapp.idx_current_module]:
        webapp.idx_current_module = appModules.index(module_selection)
        st.experimental_rerun()

    data_source_selection = mp_selectors[1].selectbox(
        "Data/Media Source:",
        appSources,
        index=webapp.idx_data_source,
    )
    if data_source_selection != appSources[webapp.idx_data_source]:
        webapp.idx_data_source = appSources.index(data_source_selection)
        st.experimental_rerun()

    st.sidebar.write("")
    ph_variables = st.sidebar.columns([1, 1])

    read_source_media(webapp, appSources, ph_variables)

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]
