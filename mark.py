path = 'test_expr'
f = open(path + '/result.txt')
m = open(path + '/mark.txt', 'w')
total = [0] * 4
correct = [0] * 4
while True:
    line = f.readline()
    if not line:
        break
    tokens = line.split()

    filename = tokens[0]
    print filename

    ans = tokens[1]
    ans = ans.replace('\\left', '')
    ans = ans.replace('\\right', '')
    if '{' in ans or '}' in ans:
        continue

    if len(tokens) > 2:
        res = tokens[2]
    width = int(filename.rstrip('.png')[-1])

    total[width] = total[width] + 1
    if ans == res:
        correct[width] = correct[width] + 1
        m.write('O')
    else:
        m.write('X')
    m.write(' %s %s %s\n' % (filename, ans, res))
f.close()
m.close()

print total
print correct
