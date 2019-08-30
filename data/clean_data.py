

filepath = "/home/name/C++2Python/data/leet_code/Python/2-keys-keyboard.py"

file = open(filepath, "r")
lines = file.readlines()
in_comment = 0

reserve_space_symbols = {'(',')','[',']',':','.',',','!','@','%'}
double_symbols = {'<','>','=','+','-','*','/','&','|'}
list_of_lines = []

for line in lines:

    line_content = line.strip()
    # remove empty lines
    if not line_content:
        continue
    # remove one-line comments
    if line_content[0] == '#':
        continue
    if line_content[0] == '"':
        in_comment = 0 if in_comment == 1 else 1
        continue

    # if the line is not a comment, print it out
    if not in_comment:
        lastone_is_double_symbols = 0
        last_index = 0
        list_of_word = []

        for i, letter in enumerate(line):
            if letter == ' ':
                if last_index != i:
                    word = ''.join(line[last_index:i])
                    list_of_word.append(word)
                    last_index = i + 1

            if letter in reserve_space_symbols:
                if last_index < i:
                    word = ''.join(line[last_index:i])
                    list_of_word.append(word)
                list_of_word.append(line[i])
                last_index = i + 1

            if letter in double_symbols:
                if lastone_is_double_symbols:
                    list_of_word[-1] = list_of_word[-1] + letter
                    lastone_is_double_symbols = 0
                else:
                    if last_index < i:
                        word = ''.join(line[last_index:i])
                        list_of_word.append(word)
                    list_of_word.append(line[i])
                last_index = i + 1

        print(list_of_word)







        #print(line.rstrip())
