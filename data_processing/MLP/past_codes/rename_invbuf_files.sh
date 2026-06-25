#!/bin/bash

# Script to rename invbuf files
# Replaces:
#   - (minus) with n (negative)
#   . (decimal point) with p (point)
#
# Examples:
#   invbuf_0_0_0_-25 -> invbuf_0_0_0_m25
#   invbuf_0_0_0_12.5 -> invbuf_0_0_0_12p5
#   invbuf_0_0_0_62.5_045.lib -> invbuf_0_0_0_62p5_045.lib

BASE_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_ex2/processed"
DRY_RUN=false  # Set to false to actually rename files

echo "🔧 File Renaming Script for invbuf files"
echo "📁 Base directory: $BASE_DIR"

if [ "$DRY_RUN" = true ]; then
    echo "⚠️  DRY RUN MODE - No files will be renamed (showing what would happen)"
    echo ""
fi

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "❌ Base directory does not exist: $BASE_DIR"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Renaming folders"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# First, rename folders
FOLDER_COUNT=0
for OLD_FOLDER in "$BASE_DIR"/invbuf_*; do
    if [ -d "$OLD_FOLDER" ]; then
        OLD_NAME=$(basename "$OLD_FOLDER")
        # Replace -25 with n25 and 12.5 with 12p5, etc.
        NEW_NAME=$(echo "$OLD_NAME" | sed 's/-/m/g' | sed 's/\./p/g')
        
        if [ "$OLD_NAME" != "$NEW_NAME" ]; then
            NEW_FOLDER="$BASE_DIR/$NEW_NAME"
            echo "📂 $OLD_NAME -> $NEW_NAME"
            
            if [ "$DRY_RUN" = false ]; then
                mv "$OLD_FOLDER" "$NEW_FOLDER"
            fi
            ((FOLDER_COUNT++))
        fi
    fi
done

echo "📊 Folders to rename: $FOLDER_COUNT"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Renaming .lib files inside folders"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Then, rename files inside each folder
FILE_COUNT=0
FOLDERS_PROCESSED=0

# Get updated folder list (use new names if already renamed)
if [ "$DRY_RUN" = true ]; then
    # In dry run, use original folder names
    FOLDERS=$(find "$BASE_DIR" -maxdepth 1 -type d -name "invbuf_*" | sort)
else
    # After renaming, look for both patterns
    FOLDERS=$(find "$BASE_DIR" -maxdepth 1 -type d -name "invbuf_*" | sort)
fi

for FOLDER in $FOLDERS; do
    FOLDER_NAME=$(basename "$FOLDER")
    FILES_IN_FOLDER=0
    
    echo "📁 Processing folder: $FOLDER_NAME"
    
    # Find all .lib files in this folder
    for OLD_FILE in "$FOLDER"/*.lib; do
        if [ -f "$OLD_FILE" ]; then
            OLD_FILENAME=$(basename "$OLD_FILE")
            # Replace -25 with m25 and 12.5 with 12p5 in filename (but preserve .lib extension)
            # First, separate the filename and extension
            FILENAME_WITHOUT_EXT="${OLD_FILENAME%.lib}"
            # Replace - with m and . with p only in the filename part
            NEW_FILENAME_WITHOUT_EXT=$(echo "$FILENAME_WITHOUT_EXT" | sed 's/-/m/g' | sed 's/\./p/g')
            # Add back the .lib extension
            NEW_FILENAME="${NEW_FILENAME_WITHOUT_EXT}.lib"
            
            if [ "$OLD_FILENAME" != "$NEW_FILENAME" ]; then
                NEW_FILE="$FOLDER/$NEW_FILENAME"
                echo "   📄 $OLD_FILENAME -> $NEW_FILENAME"
                
                if [ "$DRY_RUN" = false ]; then
                    mv "$OLD_FILE" "$NEW_FILE"
                fi
                ((FILE_COUNT++))
                ((FILES_IN_FOLDER++))
            fi
        fi
    done
    
    if [ $FILES_IN_FOLDER -gt 0 ]; then
        echo "   ✅ Files to rename in this folder: $FILES_IN_FOLDER"
        ((FOLDERS_PROCESSED++))
    else
        echo "   ℹ️  No files need renaming in this folder"
    fi
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Summary:"
echo "   Folders to rename: $FOLDER_COUNT"
echo "   Files to rename: $FILE_COUNT"
echo "   Folders processed: $FOLDERS_PROCESSED"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "⚠️  This was a DRY RUN - no files were actually renamed"
    echo "💡 To perform actual renaming, edit this script and set DRY_RUN=false"
else
    echo ""
    echo "✅ Renaming completed successfully!"
fi

echo ""
echo "📝 Examples of new naming convention:"
echo "   invbuf_0_0_0_-25 -> invbuf_0_0_0_m25"
echo "   invbuf_0_0_0_12.5 -> invbuf_0_0_0_12p5"
echo "   invbuf_0_0_0_62.5_045.lib -> invbuf_0_0_0_62p5_045.lib"