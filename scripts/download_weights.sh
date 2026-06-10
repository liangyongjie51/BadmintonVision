#!/usr/bin/env bash
# Download released model weights into ./weights
# Replace <RELEASE_URL> with the GitHub Releases asset URLs upon publication.
set -euo pipefail
mkdir -p weights
echo "Downloading pre-trained weights..."
# curl -L -o weights/ssl_convnextv2_mae_encoder.pt  <RELEASE_URL>/ssl_convnextv2_mae_encoder.pt
# curl -L -o weights/yolov12m_badminton.pt           <RELEASE_URL>/yolov12m_badminton.pt
# curl -L -o weights/motionformer.pt                 <RELEASE_URL>/motionformer.pt
echo "Edit this script with the Releases URLs (printed in the paper) before running."
