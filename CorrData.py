# A class to work with data table

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from math import log
import os
from CorrDataPath import DataPath

class CorrData():
    def __init__(self):
        csv_filename = os.path.join(DataPath, "DIB_Measruements.csv")
        self.df = pd.read_csv(csv_filename)
        header = self.df.columns[2:77:3]
        self.sightlines = [item[0:-2] for item in header]
        self.label_all = np.array(self.df.wave_label.to_list())
        wavelength = self.df.wavelength.iloc[7:].to_list()
        wave_label = self.df.wave_label.iloc[7:].to_list()
        self.wave_all = np.array([float(item) for item in wavelength])
        self.wave_label = np.array(wave_label)

    def _parseWavelength(self, wave_in):
        wave_str = str(wave_in)
        try:
            wave_float = float(wave_in)
        except:
            wave_float = None
        if wave_str in self.label_all:
            return wave_str
        elif wave_float is not None:
            idx1 = self.wave_all >= wave_float - 10.0
            idx2 = self.wave_all <= wave_float + 10.0
            if sum(idx1 * idx2) == 0:
                print(wave_str + ": Check wavelength, cannot find any matches")
                return None
            else:
                poss_wave = self.wave_all[idx1 * idx2]
                poss_label = self.wave_label[idx1 * idx2]
                print(wave_str + ": Cannot find exact matches, possible DIBs within 10 AA are:")
                for item1, item2 in zip(poss_label, poss_wave):
                    print(" @ ".join([item1, str(item2)]))
                new_label = input("Type 4-digit label")
                return self._parseWavelength(new_label)
        else:
            print(wave_str + ": a typo of parameter names?")
            return None

    def getSingleData(self, wave_in, allowed_labels="D", normalized=True):
        wave_str = self._parseWavelength(wave_in)
        if wave_str is None:
            print("No data returned")
            return None
        else:
            allowed_labels = [item.upper() for item in allowed_labels]
            data_slice = self.df.loc[self.df.wave_label == wave_str]
            ebv_slice = self.df.loc[self.df.wave_label == "Ebv"]
            for sightline in self.sightlines:
                if data_slice.iloc[0][sightline+"label"] in allowed_labels:
                    if normalized:
                        df_add = pd.DataFrame([[sightline,
                                                data_slice.iloc[0][sightline + "EW"]/ebv_slice.iloc[0][sightline + "EW"],
                                                data_slice.iloc[0][sightline + "err"],
                                                data_slice.iloc[0][sightline + "label"]
                                                ]],
                                              columns=["sightline", "EW", "err", "label"])
                    else:
                        df_add = pd.DataFrame([[sightline,
                                                data_slice.iloc[0][sightline + "EW"],
                                                data_slice.iloc[0][sightline + "err"],
                                                data_slice.iloc[0][sightline + "label"]
                                                ]],
                                              columns=["sightline", "EW", "err", "label"])
                    try:
                        df_out = df_out.append(df_add, ignore_index=True)
                    except:
                        df_out = df_add

            return df_out

    def _getDoubleData_DIB(self, wave1, wave2, allowed_labels1="D", allowed_labels2="D", normalized=True):
        allowed_labels1 = [item.upper() for item in allowed_labels1]
        allowed_labels2 = [item.upper() for item in allowed_labels2]
        data_slice1 = self.df.loc[self.df.wave_label == wave1]
        data_slice2 = self.df.loc[self.df.wave_label == wave2]
        ebv_slice = self.df.loc[self.df.wave_label == "Ebv"]
        for sightline in self.sightlines:
            if data_slice1.iloc[0][sightline + "label"] in allowed_labels1 and \
                    data_slice2.iloc[0][sightline + "label"] in allowed_labels2:

                value = [sightline]
                for data_slice in [data_slice1, data_slice2]:
                    if normalized:
                        value.append(data_slice.iloc[0][sightline + "EW"] / ebv_slice.iloc[0][sightline + "EW"])
                        value.append(data_slice.iloc[0][sightline + "err"] / ebv_slice.iloc[0][sightline + "EW"] +
                                 data_slice.iloc[0][sightline + "EW"]*0.03 / (ebv_slice.iloc[0][sightline+"EW"])**2)
                        value.append(data_slice.iloc[0][sightline + "label"])
                    else:
                        value.append(data_slice.iloc[0][sightline + "EW"])
                        value.append(data_slice.iloc[0][sightline + "err"])
                        value.append(data_slice.iloc[0][sightline + "label"])
                df_add = pd.DataFrame([value], columns=["sightline", "EW1", "err1", "label1", "EW2", "err2", "label2"])

                try:
                    df_out = df_out.append(df_add, ignore_index=True)
                except:
                    df_out = df_add

        return df_out

    def _getDoubleData_ISM_Linear(self, wave1, wave2, allowed_labels1="D", allowed_labels2="D", normalized=True):
        ISM_dict = {"NH":20, "NH2": 20, "NCN":12, "NCH":13, "NC2":13}

        allowed_labels1 = [item.upper() for item in allowed_labels1]
        allowed_labels2 = [item.upper() for item in allowed_labels2]
        data_slice1 = self.df.loc[self.df.wave_label == wave1]
        data_slice2 = self.df.loc[self.df.wave_label == wave2]
        ebv_slice = self.df.loc[self.df.wave_label == "Ebv"]
        for sightline in self.sightlines:
            if data_slice1.iloc[0][sightline + "label"] in allowed_labels1 and \
                    data_slice2.iloc[0][sightline + "label"] in allowed_labels2:
                value = [sightline]
                for data_slice, wave in zip([data_slice1, data_slice2], [wave1, wave2]):
                    EW, err = data_slice.iloc[0][sightline + "EW"], data_slice.iloc[0][sightline + "err"]
                    if wave in ISM_dict.keys():
                        err = 10 ** (EW + err - ISM_dict[wave]) - 10 ** (EW - ISM_dict[wave])
                        EW = 10 ** (EW - ISM_dict[wave])
                    if normalized:
                        err = err / ebv_slice.iloc[0][sightline + "EW"] + \
                              EW * 0.03 / (ebv_slice.iloc[0][sightline+"EW"])**2
                        EW = EW / ebv_slice.iloc[0][sightline + "EW"]
                    value.append(EW)
                    value.append(err)
                    value.append(data_slice.iloc[0][sightline + "label"])

                df_add = pd.DataFrame([value], columns=["sightline", "EW1", "err1", "label1", "EW2", "err2", "label2"])

                try:
                    df_out = df_out.append(df_add, ignore_index=True)
                except:
                    df_out = df_add

        return df_out

    def _getDoubleData_ISM_Log(self, wave1, wave2, allowed_labels1="D", allowed_labels2="D", normalized=True):
        ISM_dict = {"NH": 20, "NH2": 20, "NCN": 12, "NCH": 13, "NC2": 13}

        allowed_labels1 = [item.upper() for item in allowed_labels1]
        allowed_labels2 = [item.upper() for item in allowed_labels2]
        data_slice1 = self.df.loc[self.df.wave_label == wave1]
        data_slice2 = self.df.loc[self.df.wave_label == wave2]
        ebv_slice = self.df.loc[self.df.wave_label == "Ebv"]
        for sightline in self.sightlines:
            if data_slice1.iloc[0][sightline + "label"] in allowed_labels1 and \
                    data_slice2.iloc[0][sightline + "label"] in allowed_labels2:
                value = [sightline]
                for data_slice, wave in zip([data_slice1, data_slice2], [wave1, wave2]):
                    EW, err = data_slice.iloc[0][sightline + "EW"], data_slice.iloc[0][sightline + "err"]
                    if wave not in ISM_dict.keys():
                        err = log(EW + err, 10) - log(EW, 10)
                        EW = log(EW, 10)
                    if normalized:
                        err = err + log(ebv_slice.iloc[0][sightline + "EW"]+0.03, 10) \
                              - log(ebv_slice.iloc[0][sightline + "EW"], 10)
                        EW = EW - log(ebv_slice.iloc[0][sightline + "EW"], 10)
                    value.append(EW)
                    value.append(err)
                    value.append(data_slice.iloc[0][sightline + "label"])

                df_add = pd.DataFrame([value], columns=["sightline", "EW1", "err1", "label1", "EW2", "err2", "label2"])

                try:
                    df_out = df_out.append(df_add, ignore_index=True)
                except:
                    df_out = df_add

        return df_out

    def getDoubleData(self, wave1, wave2, allowed_labels1="D", allowed_labels2="D", normalized=True, mode="linear"):
        wave1 = self._parseWavelength(wave1)
        wave2 = self._parseWavelength(wave2)
        if wave1 is None or wave2 is None:
            print("No data returned")
            return None
        else:
            if wave1 not in ["NH", "NH2", "NCH", "NCN", "NC2"] and wave2 not in ["NH", "NH2", "NCH", "NCN", "NC2"]:
                df_out = self._getDoubleData_DIB(wave1, wave2,
                                                 allowed_labels1=allowed_labels1,
                                                 allowed_labels2=allowed_labels2,
                                                 normalized=normalized)
            else:
                if mode.lower() == "linear":
                    df_out = self._getDoubleData_ISM_Linear(wave1, wave2,
                                                 allowed_labels1=allowed_labels1,
                                                 allowed_labels2=allowed_labels2,
                                                 normalized=normalized)
                if mode.lower() == "log":
                    df_out = self._getDoubleData_ISM_Log(wave1, wave2,
                                                         allowed_labels1=allowed_labels1,
                                                         allowed_labels2=allowed_labels2,
                                                         normalized=normalized)


            return df_out

    def getPearsonr(self,wave1, wave2, allowed_labels1="D", allowed_labels2="D", normalized=True, mode="linear"):
        df = self.getDoubleData(wave1, wave2, allowed_labels1, allowed_labels2, normalized, mode=mode)
        r, _ = pearsonr(df.EW1.to_list(), df.EW2.to_list())
        return r

    def get_fH2_single(self, wave_in, allowed_labels="D", normalized=True):
        wave_str = self._parseWavelength(wave_in)
        if wave_str is None:
            print("No data returned")
            return None
        else:
            allowed_labels = [item.upper() for item in allowed_labels]
            data_slice = self.df.loc[self.df.wave_label == wave_str]
            ebv_slice = self.df.loc[self.df.wave_label == "Ebv"]
            fH2_slice = self.df.loc[self.df.wave_]
            for sightline in self.sightlines:
                if data_slice.iloc[0][sightline + "label"] in allowed_labels:
                    if normalized:
                        df_add = pd.DataFrame([[sightline,
                                                data_slice.iloc[0][sightline + "EW"] / ebv_slice.iloc[0][
                                                    sightline + "EW"],
                                                data_slice.iloc[0][sightline + "err"],
                                                data_slice.iloc[0][sightline + "label"]
                                                ]],
                                              columns=["sightline", "EW", "err", "label"])
                    else:
                        df_add = pd.DataFrame([[sightline,
                                                data_slice.iloc[0][sightline + "EW"],
                                                data_slice.iloc[0][sightline + "err"],
                                                data_slice.iloc[0][sightline + "label"]
                                                ]],
                                              columns=["sightline", "EW", "err", "label"])
                    try:
                        df_out = df_out.append(df_add, ignore_index=True)
                    except:
                        df_out = df_add

if __name__ == "__main__":
    dataset = CorrData()

    df = dataset.getDoubleData(wave1=5780, wave2="NH", allowed_labels2=["D", "S"], normalized=True, mode="linear")
    r = dataset.getPearsonr(wave1=5780, wave2="NH", allowed_labels2=["D", "S"], normalized=True, mode="linear")
    for x, y, err_x, err_y in zip(df.EW1.to_list(), df.EW2.to_list(), df.err1.to_list(), df.err2.to_list()):
        plt.scatter(x, y, color="k")
        plt.plot([x-err_x, x+err_x], [y,y], color="k")
        plt.plot([x,x], [y-err_y, y+err_y], color="k")
    plt.xlabel("W(5780)/Ebv")
    plt.ylabel("N(H)/Ebv")
    plt.savefig("/Users/haoyufan/test.png")
    #plt.show()
    print("r for normalized: {r:.2f}".format(r=r))



