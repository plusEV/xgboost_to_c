mport contextlib
from sklearn.tree.tree import DecisionTreeRegressor, DTYPE
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble.forest import ForestRegressor
#ALWAYS_INLINE = "__attribute__((__always_inline__))"

ALWAYS_INLINE = "ALWAYS_INLINE"

class CodeGenerator(object):
    def __init__(self):
        self._lines = []
        self._indent = 0

    @property
    def lines(self):
        return self._lines

    def write(self, line):
        self._lines.append("  " * self._indent + line)

    @contextlib.contextmanager
    def bracketed(self, preamble, postamble):
        assert self._indent >= 0
        self.write(preamble)
        self._indent += 1
        yield
        self._indent -= 1
        self.write(postamble)

def code_gen_tree(tree,fn,gen=None):
    if gen is None:
        gen = CodeGenerator()

    def recur(ttree,level=0):
        result = {}
        for i in range(0,len(ttree)):
            cn = ttree[i]

            try:
                nn  = ttree[i+1]
            except:
                nn = {'level':-1}

            if cn['level']>level:
                continue
            if cn['level']<level:
                return

            branch = "if ({0}f) {{".format(cn['line'])

            if nn['level']==level:
                gen.write("return {0}f;".format(cn['line']))
            elif nn['level']>level:
                with gen.bracketed(branch,"}"):
                    recur(ttree[i+1:],level=nn['level'])
            else:
                with gen.bracketed("else {", "}"):
                    gen.write("return {0}f;".format(cn['line']))

    fn_decl = "{inline} double {name}(double* f) {{".format(
        inline=ALWAYS_INLINE,
        name=fn)
    
    info = []
    for line in lines[1:]:
        line = line.replace('        ','\t')
        level = line.count('\t')
        s = line.split(',')[0].replace('\t','')[2:]
        if s[:4] == 'leaf':
            s=s[5:]
        else:
            d = s.find('<')
            n = s[2:d]
            s = s[1] + '[' + n + ']' + s[d:s.find(']')]
        info.append({'line': s,'level' : level})
        
    with gen.bracketed(fn_decl, "}"):
        recur(info)
    return gen.lines

def get_tree(it):
    tree = []
    while True:
        line = next(it,'end')
        if re.search('booster',line):
            if tree:
                yield tree
                tree=  []
        elif line == 'end':
            yield tree
            break
        else:
            tree.append(line)

def code_gen_ensemble(model_path,fn,gen=None):
    if gen is None:
        gen = CodeGenerator()
    it = open(model_path)
    
    num_trees = 0
    for i, tree in enumerate(get_tree(it)):
        name = "{name}_{index}".format(name='boost', index=i)
        code_gen_tree(tree,name, gen)
        num_trees+=1
        
    fn_decl = "double {name}(double* f) {{".format(name=fn)
    with gen.bracketed(fn_decl, "}"):
        gen.write("double result = 0.;")
        for i in range(num_trees):
            increment = "result += {name}_{index}(f);".format(
                name='boost',index=i)
            gen.write(increment)
        gen.write("return result;")
    return gen.lines

def xgb_to_c(model_path,fn):
    lines = code_gen_ensemble(model_path,fn=fn)
    assert lines is not None
    return "\n".join(lines)

