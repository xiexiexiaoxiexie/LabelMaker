def read_mask_file(file_path):
    """
    Reads a mask file and returns a list of integers (0 or 1).
    """
    with open(file_path, 'r') as f:
        return [int(line.strip()) for line in f]

def check_overlap(mask1, mask2):
    """
    Checks if the two masks overlap (i.e., both have 1 at the same index).
    Returns a list of indices where overlap occurs.
    """
    overlap_indices = []
    for i in range(len(mask1)):
        if mask1[i] == 1 and mask2[i] == 1:
            overlap_indices.append(i)
    return overlap_indices

def main(mask_file1, mask_file2):
    # Read the two mask files
    mask1 = read_mask_file(mask_file1)
    mask2 = read_mask_file(mask_file2)

    # Check if the two masks are of the same length
    if len(mask1) != len(mask2):
        print("The mask files have different lengths!")
        return

    # Check for overlap
    overlap_indices = check_overlap(mask1, mask2)

    if overlap_indices:
        print(f"Overlap found at indices: {overlap_indices}")
    else:
        print("No overlap found between the two masks.")

if __name__ == "__main__":
    folder='/home/xiefujing/LabelMaker/apple_scanner/2024_11_15_14_10_27_office/output/intermediate/scannet200_mask3d_1/pred_mask/'
    # Example file paths, replace with actual paths to your mask files
    mask_file1 = folder+'006.txt'
    mask_file2 = folder+'003.txt'
    
    main(mask_file1, mask_file2)
