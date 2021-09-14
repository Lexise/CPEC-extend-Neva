import clingo
import os

def compute_extensions(apx_file, asp_file, extension_file):
    ctl = clingo.Control("0")
    # with open("data/app_uploaded_files/n100p3q34ve.apx", 'r') as file:
    #     apx = file.read()
    # with open("prefex.dl", 'r') as file:
    #     asp = file.read()
    # ctl.add('base', [], '''\
    #      p(1).
    #     { p(3) }.
    #     #external p(1..3).
    #
    #     q(X) :- p(X).
    #     ''')
    #ctl.add('base',[], apx+asp)

    ctl.load(apx_file) # eg."data/app_uploaded_files/n100p3q34ve.apx
    ctl.load(asp_file)# eg."prefex.dl")

    ctl.ground([("base", [])])

    file1 = open(extension_file,    "w")

    with ctl.solve(yield_=True) as handle:
        n=1
        for m in handle:
            file1.writelines(("Answer: {}\n{}\n".format(n,m)))
            if round(os.path.getsize(extension_file)/(1024*1024),3) > 10: #Megabytes
                print("size:",os.path.getsize(extension_file))
                ctl.interrupt()
            n+=1
            handle.get()

    file1.close()
