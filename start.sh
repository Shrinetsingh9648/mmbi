#!/bin/bash
# Create a symlink so Python can resolve 'from mmbi import ...'
ln -sfn /opt/render/project/src /opt/render/project/mmbi

# Tell Python where to find it
export PYTHONPATH=/opt/render/project

# Start the server
python -m mmbi.server
