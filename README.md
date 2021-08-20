<p align="right">
<img src="https://badges.pufler.dev/visits/Outsiders17711/streamlit-Mediapipe-WebApp?style=for-the-badge&logo=github" alt="https://github.com/Outsiders17711" />&nbsp;
<img src="https://badges.pufler.dev/updated/Outsiders17711/streamlit-Mediapipe-WebApp?style=for-the-badge&logo=github" alt="https://github.com/Outsiders17711" />&nbsp;
<img src="https://badges.pufler.dev/created/Outsiders17711/streamlit-Mediapipe-WebApp?style=for-the-badge&logo=github" alt="https://github.com/Outsiders17711" />&nbsp;
</p>

## Streamlit Mediapipe WebApp

This application explores the functionality of some of Google's <a href="https://google.github.io/mediapipe/" style="text-decoration: none;">**Mediapipe**</a> Machine Learning solutions, viz:

  - Hand Tracking
  - Pose Estimation
  - Face Detection 
  - Face Mesh
    
<a href="https://streamlit.io/" style="text-decoration: none;">**StreamLit**</a> is used to create the Web Graphical User Interface (GUI). Streamlit is a cool way to turn data scripts into shareable web apps in minutes, all in Python. You can check out the launch <a href="https://towardsdatascience.com/coding-ml-tools-like-you-code-ml-models-ddba3357eace" style="text-decoration: none;">*blog post*</a> for more information.

This web app was inspired by the awesome YouTube tutorials created by:

- **Murtaza Hassan**: <a href="https://www.youtube.com/watch?v=01sAkU_NvOY" style="text-decoration: none;">Advanced Computer Vision with Python - Full Course</a>
- **Augmented Startups**: <a href="https://www.youtube.com/watch?v=wyWmWaXapmI" style="text-decoration: none;">StreamLit Computer Vision User Interface Course | MediaPipe OpenCV Python (2021)</a>

Do check out their channels and websites for more informative and exciting Machine Learning & Computer Vision tutorials.

---


## Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/outsiders17711/streamlit-mediapipe-webapp/main/streamlitMediapipeOnline.py)


The app was been deployed on Streamlit. You can check it out **[here](https://share.streamlit.io/outsiders17711/streamlit-mediapipe-webapp/main/streamlitMediapipeOnline.py)**.

![Short Demo](https://github.com/Outsiders17711/streamlit-Mediapipe-WebApp/blob/main/demo/streamlitMediapipe.gif?raw=true)

### Note: Video Sources Disabled On Streamlit Share

There have been a couple of issues running video sources on the online shared app:

1. There is a considerable lag when running the Mediapipe modules on videos with one viewer/user. 
2.  If multiple viewers/users attempt to use video inputs at the same time, the app becomes totally unresponsive and needs to be rebooted.

I suspect that both issues are related but I have been unable to get them fixed. Kindly reach out if you have any ideas/suggestions on how I can go about resolving these issues.


**If you want to test out video sources, you can clone the [source repository](https://github.com/Outsiders17711/streamlit-Mediapipe-WebApp), install the requirements and run `streamlitMediapipe.py`.**


<hr>
