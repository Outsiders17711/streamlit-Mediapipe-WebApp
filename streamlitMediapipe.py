import dataclasses

import streamlit as st

from functions import *


# [start] [persistent states]__________________________________________
@dataclasses.dataclass
class webappState:
    idx_current_page: int = 1
    current_page: str = "About Web App"
    page_selector_key: int = 0

    idx_current_module: int = 0
    current_module: str = "Hand Tracking"
    module_selector_key: int = 0

    idx_data_source: int = 1
    data_source: str = "Random Image"
    source_selector_key: int = 0


@st.cache(allow_output_mutation=True)
def _webappState() -> webappState:
    return webappState()


webapp = _webappState()
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

# [start] [setup main page and side bar] ____________________________
st.markdown(
    f"""
    {pageConfig}
    <div style="text-align:center; margin-top:-75px; ">
    <h1 style="font-variant: small-caps; font-size: xx-large; margin-bottom:-45px;" >
    <font color=#ea0525>w e b {nbsp*2} a p p</font>
    </h1>
    <h1> ADVANCED COMPUTER VISION WITH PYTHON </h1>
    <hr>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"""
    <div style="text-align:center; margin-top:-75px; ">
    <h3 style="font-variant: small-caps; font-size: xx-large; ">
    <font color=#ea0525>s i d e {nbsp} b a r</font>
    </h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

# [start] [setup app pages, modules & data sources]__________________

appPages = ["Mediapipe Modules", "About Web App", "About Me"]
page_selection = st.sidebar.selectbox(
    "Page Selection:",
    appPages,
    index=webapp.idx_current_page,
)
if webapp.current_page != page_selection:
    webapp.idx_current_page = appPages.index(page_selection)
    webapp.current_page = page_selection
    st.experimental_rerun()

if webapp.current_page == "About Me":
    st.markdown(aboutMe(), unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.image(open_img_path_url("highlord.jpg", "path"), use_column_width="auto")
    st.sidebar.markdown("---")

elif webapp.current_page == "About Web App":
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

    vid1, vid2 = st.beta_columns([1, 1])
    vid1.video("https://www.youtube.com/watch?v=01sAkU_NvOY")
    vid1.caption("Advanced Computer Vision with Python - Full Course")
    vid2.video("https://www.youtube.com/watch?v=wyWmWaXapmI")
    vid2.caption(
        "StreamLit Computer Vision User Interface Course | MediaPipe OpenCV Python (2021)"
    )

    st.markdown(aboutWebApp()[1], unsafe_allow_html=True)

elif webapp.current_page == "Mediapipe Modules":
    st.set_option("deprecation.showfileUploaderEncoding", False)

    mp_selectors = st.sidebar.beta_columns([1, 1])

    appModules = ["Hand Tracking", "Pose Estimation", "Face Detection", "Face Mesh"]
    module_selection = mp_selectors[0].selectbox(
        "Choose The Mediapipe Solution:",
        appModules,
        index=webapp.idx_current_module,
    )
    if webapp.current_module != module_selection:
        webapp.idx_current_module = appModules.index(module_selection)
        webapp.current_module = module_selection
        st.experimental_rerun()

    appDataSources = [
        "User Image",
        "Random Image",
        "User Video",
        "Random Video",
        "WebCam",
    ]
    data_source_selection = mp_selectors[1].selectbox(
        "Select Media Source:",
        appDataSources,
        index=webapp.idx_data_source,
    )
    if webapp.data_source != data_source_selection:
        webapp.idx_data_source = appDataSources.index(data_source_selection)
        webapp.data_source = data_source_selection
        st.experimental_rerun()

    st.sidebar.markdown("")
    ph_variables = st.sidebar.beta_columns([1, 1])
    st.sidebar.markdown("")

    media, type = read_source_media(webapp.data_source)
    run_selected_module(webapp.current_module, media, type, ph_variables)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]
