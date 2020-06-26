def index_to_ints(index):
    str1 = index
    x = None
    y = None

    try:
        x = int(str1[1] + str1[2])
    except:
        try: 
            x = int(str1[1])
        except:
            print("Fail")

    try: 
        y = int(str1[5] + str1[6])
    except:
        try:
            y = int(str1[4] + str1[5])   
        except:
            try:
                y = int(str1[4])
            except:
                try:
                    y = int(str1[5])
                except:
                    print("ne")


    return x, y

def distance(px, py, gx, gy):
    distance = abs(px - gx) + abs(py - gy)

    return distance