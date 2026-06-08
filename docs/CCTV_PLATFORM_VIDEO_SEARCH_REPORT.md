# CCTV Platform Video Search Report

## Goal

The previous test video looked closer to phone-angle footage than station CCTV. The new goal was to find real railway, metro, or train-station footage with a CCTV-like elevated/static angle, a wide platform view, visible passengers at distance, and enough crowding to test full-body detection versus head detection.

## Candidate Videos

| rank | title/name | URL | source | railway/metro/platform specific | CCTV-like angle | crowd level | downloadable | license/usage note | score | notes |
|---:|---|---|---|---|---|---|---|---|---:|---|
| 1 | People on Platform on Train Station | https://www.pexels.com/video/people-on-platform-on-train-station-12049569/ | Pexels | yes | partial - elevated fixed-looking platform shot, not actual CCTV | moderate | yes, direct MP4 found | Pexels page says Free to use / Free download | 4.5 | Best practical test clip: real platform, train present, people at several distances, good for body/head comparison. |
| 2 | Crowded Train Station | https://www.pexels.com/video/crowded-train-station-6023186/ | Pexels | yes | partial - station/platform stock angle | moderate | yes, direct MP4 found | Pexels page says Free to use / Free download | 4.0 | Useful railway-station crowd clip, but described as timelapse, which is less ideal for tracking stability. |
| 3 | Time-Lapse Video of People inside a Metro Station | https://www.pexels.com/video/time-lapse-video-of-people-inside-a-metro-station-5097500/ | Pexels | metro station, platform-related | yes - high angle / overhead | high | likely yes, page lists free download | Pexels page says Free to use / Free download | 3.8 | Strong CCTV-like view and crowding, but timelapse and station interior may reduce tracking realism. |
| 4 | A View of the Metro Station | https://www.pexels.com/video/a-view-of-the-metro-station-5921058/ | Pexels | yes | partial - station/platform, but likely moving/dolly shot | low to moderate | yes, direct MP4 found | Pexels page says Free to use / Free download | 3.5 | Real metro platform setting, but less crowded and less CCTV-like than rank 1. |
| 5 | People Walking on a Subway Station | https://www.pexels.com/video/people-walking-on-a-subway-station-6145943/ | Pexels | yes | no to partial - closer/vertical stock shot | low to moderate | likely yes, page lists free download | Pexels page says Free to use / Free download | 3.0 | Platform passengers are visible, but the angle is less suitable for calibrated zone/line testing. |
| 6 | MetroStation Dataset | https://figshare.com/articles/dataset/MetroStation_Dataset/20521848/1 | Figshare | yes | yes - surveillance-derived metro scenes | varied | yes, dataset download | CC0 on Figshare | 3.0 | Very relevant for passenger detection, stairs/escalator/gate/platform scenes, but it contains annotated images rather than a processable video clip. |
| 7 | Grand Central Station Dataset | https://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html | CUHK | train station, not platform | yes - fixed elevated station view | high | yes, 1.1 GB AVI via FTP | public research dataset; license not explicit, citation requested | 3.0 | Excellent true crowd analytics benchmark, but it is a concourse rather than platform and the license is less explicit than Pexels/CC0. |
| 8 | File:Gct.ogv | https://commons.wikimedia.org/wiki/File:Gct.ogv | Wikimedia Commons | train station, not platform | no to partial | moderate | yes | Wikimedia page lists Creative Commons terms | 2.5 | Real Grand Central interior video with clear reuse terms, but the viewpoint is not CCTV-like and not platform-specific. |
| 9 | Commuters at New York's Grand Central Station | https://mixkit.co/free-stock-video/commuters-at-new-yorks-grand-central-station-4180/ | Mixkit | train station, not platform | no to partial | moderate | yes | Mixkit Stock Video Free License | 2.5 | Legal/free source, but more cinematic concourse footage than surveillance/platform footage. |
| 10 | Crowded Nairobi Railway Station | https://www.videvo.net/video/crowded-nairobi-railway-station/616199/ | Videvo | yes | unclear | moderate | possibly, but premium/editorial constraints shown | royalty-free/editorial notes; not the cleanest automated download | 2.5 | Relevant station/platform content, but less convenient and usage/download terms are less clean for this workflow. |

## Selected Video

- selected video name: People on Platform on Train Station
- source URL: https://www.pexels.com/video/people-on-platform-on-train-station-12049569/
- canonical sample path: data/input_videos/sample.mp4
- preserved copy: data/input_videos/cctv_platform_sample.mp4
- source metadata: data/input_videos/sample.source.json
- why selected: it is real railway platform footage with an elevated, static-looking angle, a train at the platform, multiple passengers, visible people at near and far distances, and enough occlusion/crowding to compare full-body detection with head detection.
- whether it is real footage: yes, it appears to be real recorded platform footage.
- whether it is CCTV-like: partially. It has an elevated fixed-looking viewpoint, but it is stock footage, not confirmed CCTV.
- whether it has crowds: yes, moderate crowding near the platform/train doors.
- limitations: not Indian Railway-specific, not confirmed CCTV, stock-footage provenance rather than an official surveillance dataset, and it may not represent the exact lens placement/noise/compression of station CCTV.

## Video Properties

- duration: 32.13 seconds
- FPS: 25
- resolution: 1280 x 720
- frame count: 803
- format size: about 10.5 MB

## Files Created

- data/input_videos/sample.mp4
- data/input_videos/sample.source.json
- data/input_videos/cctv_platform_sample.mp4
- data/input_videos/cctv_platform_sample.source.json
- data/sample_frames/cctv_platform_sample/

The sample frame directory contains one extracted JPG per second from the selected clip, plus a representative preview frame.

## Usage Notes

- This video is for testing/demo only.
- It is not training data unless it is separately labeled and the license/use case is rechecked for training.
- It is not necessarily Indian railway-specific.
- It should not be described as actual CCTV. It is best described as CCTV-like elevated railway platform footage.
- Final validation still needs approved real Indian Railway CCTV footage or a formally released Indian railway/metro CCTV dataset.

## Recommendation For Next Agent

This video has been made the canonical `data/input_videos/sample.mp4` for the next local demo review. It is substantially better than phone-angle footage for testing zone occupancy, line-crossing overlays, and body-vs-head detection behavior.

Do not present it as real CCTV. The correct phrasing is: real railway platform stock footage with a CCTV-like elevated/static angle. For production claims, continue searching for a formally licensed CCTV/platform dataset or use approved station footage under a privacy-safe process.
