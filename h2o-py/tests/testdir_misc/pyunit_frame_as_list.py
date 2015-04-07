import sys
sys.path.insert(1, "../../")
import h2o

def frame_as_list(ip,port):
    # Connect to h2o
    h2o.init(ip,port)

    iris = h2o.import_frame(path=h2o.locate("smalldata/iris/iris_wheader.csv"))
    prostate = h2o.import_frame(path=h2o.locate("smalldata/prostate/prostate.csv.zip"))
    airlines = h2o.import_frame(path=h2o.locate("smalldata/airlines/allyears2k.zip"))
    iris.show()
    prostate.show()
    airlines.show()

    ###################################################################

    res1 = h2o.as_list(iris)
    assert abs(res1[8][0] - 4.4) < 1e-10 and abs(res1[8][1] - 2.9) < 1e-10 and abs(res1[8][2] - 1.4) < 1e-10, \
        "incorrect values"

    res2 = h2o.as_list(prostate)
    assert abs(res2[6][0] - 7) < 1e-10 and abs(res2[6][1] - 0) < 1e-10 and abs(res2[6][2] - 68) < 1e-10, \
        "incorrect values"

    res3 = h2o.as_list(airlines)
    assert abs(res3[3][0] - 1987) < 1e-10 and abs(res3[3][1] - 10) < 1e-10 and abs(res3[3][2] - 18) < 1e-10, \
        "incorrect values"

if __name__ == "__main__":
    h2o.run_test(sys.argv, frame_as_list)