def high_comments_count():
    file_path = 'daily_comment_count.txt'
    high_comment_counts = []  # This will store dates with comment counts of four or more digits

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  # Strip to remove whitespace and split the line into parts
            # We expect two parts: date and count. If not, we skip the line
            if len(parts) == 2:
                date, count = parts
                print(count)
                if len(count) >= 4:  # Check if the count is four digits or more
                    high_comment_counts.append(f"{date} {count}\n")

    # Write the high comment counts to a new file
    new_file_path = 'high_comment_counts.txt'
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(high_comment_counts)





def print_model_summary(model):
    for layer in model.named_modules():
        print(f"Layer: {layer[0]}")
        for param in layer[1].parameters():
            print(f"  - Parameter shape: {param.size()}")