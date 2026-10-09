# AI Usage Log

I used Gemini Flash for this project, as it balances well between coding and research.

- I used it to simplify and summarize the requirements of the README and give me a step by step plan.
- At various stages in development I asked it what I had not fulfilled from the README.
- I used it to list dependencies I needed in my conda environment.
- I used it to assist me in developing all of my scripts. 
- I used it to help me explain and comment highly technical lines.
- I used it to give suggestions for, but not write, my METHODOLOGY and AI_LOG.

### For Coding

Its proficiency at PyTorch boilerplate was excellent and I had no issues there. It did, however, repeatedly remind me to add features I already did and repeat code without my asking, so I gave it a memory instruction to assume I always reviewed its solution unless otherwise noted. Some of its code was not properly styled and that is discussed more below. 

I also changed the repository structure and filepaths from what it recommended since I did not give it full knowledge of the repo structure. I wanted it to give me code segments and not the full project. I was ultimately responsible for assembling the project, linking up API keys (none were given to it, very important) and ensuring requirements were met.

Below are some standout particular disagreements.

## 1. Flake & Black Formatting

**What I asked:**
I've always liked the idea of a clean repo and so I applied Flake and Black to this repo and made Github enforce it in a `lint.yml`. However that enforces line length and one of the lines, a formatted output string (`f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}"`) in `inference.py` could not be reduced, and I found this issue with other lines too. I asked how to resolve it.

**What I kept vs. rewrote, and why:**
It suggested that I use a multi-line string wrapping approach (`line = (f"..." f"...")`), but I decided to go with its secondary suggestion of adding inline `# noqa: E501` comments for Flake to stop flagging them.

**What the AI got wrong that I had to catch:**
The suggestion to include multi-line string wrapping was my initial preferred method, but when Black is triggered via Ctrl+S, it undoes these changes as they are not standard practice, so while it worked for Flake, it did not for Black.

**How I verified it:**
Used my `lint.yml` to check in Github that no files were being flagged for style errors.

---

## 2. Develop Branch Merging

**What I asked:**
I had been working on a separate `develop` branch rather than on `main` because that is industry practice, but I did not know if when merging, all my commits to develop would only show up as 1 commit. I queried whether development on a separate `develop` branch would preserve the full commit history upon merging back into `main` for candidate evaluation.

**What I kept vs. rewrote, and why:**
It explained that my multiple commits would only be condensed if I ran `git merge --squash` which was advised against by the project README anyway. I had never used this command and I planned to just use `git merge develop` as I usually do.

**What the AI got wrong that I had to catch:**
The AI asserted that GitHub Pull Requests default to squashing unless explicitly configured, but I do not believe this is true.

**How I verified it:**
Checked Git & GitHub documentation to confirm.

---

## 3. Scale Augmentation Decision

**What I asked:**
I asked for advice on whether to enable scale augmentationd (`scale=0.5`) and retrain the YOLOv8 model for 40 epochs on CPU vs. submitting the existing model (**0.9115 mAP50**) with scale documented as a future improvement.

**What I kept vs. rewrote, and why:**
I kept its recommendation to prioritize completion of required deliverables over re-training. I added scale augmentation to the limitations part of methodology.md in order to document that I had considered adding it.

**What the AI got wrong that I had to catch:**
When asking for what other fields I could add to the YOLO parameters, AI incorrectly assumed in that scale jittering had already been applied during training, while I knew I had not added it. I caught this by auditing the actual `model.train()` parameters (`hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `degrees=15.0`, `fliplr=0.5`), where `scale` was omitted and thus defaulted to `0.0`.

**How I verified it:**
Inspected the printed training hyperparameter output log from `model.train()` to confirm `scale=0.0` was passed into the trainer engine during the 3.0-hour training run.

---

## 4. Power & FPS for Limitations Section

**What I asked:**
I asked how to incorporate total power consumption and inference latency into the limitations section of my methodology report.

**What I kept vs. rewrote, and why:**
I kept the energy calculation breakdown (I got my Wh usage from my battery storage, power use metrics, and percentage dropoff over training time) as I did not know how to calculate that myself. I learned about RKNN and kept its quantization explanation.

**What the AI got wrong that I had to catch:**
It initially said my framerate at 14.5 FPS was slow, but when I researched into real rover framerates, their framerate count is more in the realm of minutes, not seconds. I had to clarify that the slow FPS would only be a problem because it was already slow on a high powered CPU which would only become more problematic at low power - not because 14.5 itself was slow for a rover.

**How I verified it:**
Cross-referenced Rockchip RKNN documentation and NASA AutoNav articles.