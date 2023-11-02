import jsonlines

# final answer
def parse_final_answer(solution):
    boxes = []
    start = 0

    while True:
        # Find the next occurrence of \boxed{...} starting from the current position
        start_match = solution.find(r'\boxed{', start)
        if start_match == -1:
            break  # No more boxes found
        
        # Find the closing } for the current box
        end = start_match + len(r'\boxed{')
        count = 1
        while count > 0:
            if solution[end] == '{':
                count += 1
            elif solution[end] == '}':
                count -= 1
            end += 1
        
        # Extract the content inside the box
        box_content = solution[start_match + len(r'\boxed{'):end - 1]
        boxes.append(box_content.strip())  # Strip leading/trailing whitespace if needed
        
        # Move the starting position to the character after the closing }
        start = end

    if len(boxes) == 0:
        # then, try to match \box without {}
        start = 0

        while True:
            # Find the next occurrence of \boxed... starting from the current position
            start_match = solution.find(r'\boxed', start)
            if start_match == -1:
                break  # No more boxes found
            
            # this pattern only allows one digit inside 
            box_content = solution[start_match + len(r'\boxed'):start_match + len(r'\boxed') + 1]
            boxes.append(box_content.strip())
            start = start_match + len(r'\boxed') + 1
        
        if len(boxes) == 0:
            return None

    # check if duplicate, if same, keep one
    old_len = len(boxes)
    boxes = list(set(boxes))
    new_len = len(boxes)
    # if old_len != new_len:
        # print("Duplicate detected!")

    return boxes


input_path = "./MATH_test.jsonl"

data = []

with open(input_path, "r", encoding="utf8") as f:
    for idx, item in enumerate(jsonlines.Reader(f)):
        solution = item['output']
        answer = parse_final_answer(solution)
        if answer is None:
            continue
        data.append({
            "problem": item["instruction"],
            "answer": ", ".join(answer),
            "idx": idx
        })

with open("./test.jsonl", "w", encoding="utf8") as f:
    writer = jsonlines.Writer(f)
    writer.write_all(data)
    writer.close()