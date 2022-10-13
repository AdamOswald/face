def replaceAll(string, start, end):
    for i in range(len(start)):
        string = string.replace(start[i], end[i])
    return string

def smartSplit(string, keys):
    a = list(filter(None,self.replaceAll(string, keys, ["ㅁ"]*len(keys)).split("ㅁ")))
    b = list(filter(None, self.replaceAll(string, list(" !."),["~"]*3).split("~")))
    res = []
    for i in range(len(a)):
        res.append([b[i],a[i].count("!")-a[i].count(".")])
    return res

def convert(code):
    if not "아시는구나! 참고로 겁나 어렵습니다" in code: print("이게 어딜봐서 샌즈 프로그래밍 언어 코드냐 ㅋㅋ루삥뽕"); return None
    code = (code + " ").replace("아시는구나! 참고로 겁나 어렵습니다","어렵습니다")
    BFEXcmd = list("><v^+-*[].,=")
    commands = "샌즈.알피스.파피루스.메타톤.와.토비폭스.겁나.차라.프리스크.언더테일.모르시는구나.어렵습니다".split(".")
    cmddict = {commands[i]:BFEXcmd[i] for i in range(len(commands))}
    res = ''.join(list(map(lambda x:cmddict[x[0]]*(x[1]+1),self.smartSplit(code,commands))))
    return res
