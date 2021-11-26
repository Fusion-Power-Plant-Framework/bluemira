# This is a sample Python script.

import os
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np

from bluemira.codes.plasmod.plasmodapi import PlasmodSolver as Plasmod

if __name__ == "__main__":

    input_file = "DEMO_2017.inp"
    output_file = "DEMO_2017.out"
    profile_file = "DEMO_2017.prof"

    # path to plasmod executable (to be placed in a private repository)
    plasmod_path = "../../../../plasmod_bluemira"

    plasmodObj = Plasmod(
        params={"R0": 8.973, "tol": 1e-6},
        input_file=input_file,
        output_file=output_file,
        profiles_file=profile_file,
    )

    # plasmodObj.write_input_file(input_file)
    plasmodObj.run()

    # run command
    # plasmodObj.run(plasmod_path, input_file, output_file, profile_file)

    plasmodObj.read_output_files(output_file, profile_file)

    # display some scalars
    print("Plasma current [MA]:", plasmodObj.plasma_current())
    print("Total fusion power [MW]:", plasmodObj.fusion_power())
    print("Total radiation power [MW]:", plasmodObj.radiation_power())

    # plot pprime and FFprime
    X = plasmodObj.norm_flux()
    FFprime = plasmodObj.get_FFrime()
    pprime = plasmodObj.get_pprime()

    fig, ax = plt.subplots()
    ax.plot(X, pprime)

    ax.set(xlabel="X (-)", ylabel="pprime (Pa/Wb)")
    ax.grid()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(X, FFprime)

    ax.set(xlabel="X (-)", ylabel="FFprime (T)")
    ax.grid()
    plt.show()

    # d = StringIO("M 21 72\nF 35 58")
    #
    # f = open(profile_file, "r")
    # lines = f.readlines()
    # result = []
    # for x in lines:
    #     result.append(x.split(' ')[0:9])
    # f.close()

    # with open(profile_file, "r") as f:
    #     for line in f:
    #         data = StringIO(line)
    #         print(data)
    #         values = np.loadtxt(line, delimiter='        ', usecols=[1, 2])
    #         #print(values)
