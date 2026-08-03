#!/bin/bash

# Read the incoming arguments
object_name=${1}
object_id=${2}

# Check that enough arguments were provided
if [ -z "$object_name" ]; then
    echo "Error: object_name is required."
    echo "Usage: $0 <object_name> [object_id]"
    exit 1
fi

# Check whether object_id is empty
if [ -z "$object_id" ]; then
    # object_id empty: pass an empty string
    python utils/generate_object_description.py "$object_name" 
else
    # object_id non-empty: pass it through
    python utils/generate_object_description.py "$object_name" --index "$object_id"
fi