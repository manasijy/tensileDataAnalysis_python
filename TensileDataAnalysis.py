# import scipy.io
import sys
import numpy as np
from numpy.linalg import inv
import pandas as pd
import h5py #to read matfile data
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

def find_closest(array, number):
    idx = np.argmin(np.abs(array - number))
    return idx
xl = pd.ExcelFile(r'C:\Users\MKY\OneDrive\PythonWork\DeepakData.xlsx')

# Load a sheet into a DataFrame by its name
df = xl.parse('Sheet1')

# Now you can access your columns and assign them to variables
start_row =2 #3
end_row =4128#4236
EStrain_series = df['C'].iloc[start_row:end_row]#C
EStrain = np.array(EStrain_series)
EStress_series = df['D'].iloc[start_row:end_row]#D
EStress = np.array(EStress_series)
EStrain = EStrain.astype(float)
EStress = EStress.astype(float)
# strs = np.array(LocalStressMatrix)
# fig, ax = plt.subplots(2,2,figsize=(5, 5), layout='constrained')
plt.plot(EStrain, EStress)
plt.xlabel('Engg Strain',fontsize=14)
plt.ylabel('Engg Stress (MPa)',fontsize=14)
plt.show()
input()
# plt.set(xlim = [0, 0.3],ylim = [0, 500]) #title = ' True Stress vs., True Strain',    
# plt.tick_params(axis='both', which='both', direction='in',labelsize=14,top=True, right=True)
# plt.xticks([0.1, 0.15,0.2,0.25, 0.3])
# plt.xticklabels(['0.1', None, '0.2',None, '0.3'])
# plt.yticks([100, 200, 300, 500])
# plt.yticklabels(['100', '200', '300', '500'])
# plt.show()
###################################################################
Upper_stress_value = 100#50
index_of_closest = np.argmin((np.abs(EStress - Upper_stress_value)))
if len(EStrain[:index_of_closest]) > 1 and len(EStress[:index_of_closest]) > 1:
  slope, intercept, r_value, p_value, std_err = stats.linregress(EStrain[:index_of_closest], EStress[:index_of_closest])   
else:
    print("E[:index] and F[:index] need to be arrays with more than one element.")
index_of_YS = np.argmin((np.abs(EStress - slope*(EStrain-0.002)-intercept)))
YS = EStress[index_of_YS]
x=np.linspace(0,0.05,50)
plt.plot(EStrain, EStress)
plt.plot(x,slope*(x-0.002)+intercept)
plt.show()
###################################################################
indx = index_of_YS
maxStressIndex = np.where(EStress == max(EStress))
strain = np.log(1+EStrain[indx:maxStressIndex[0][0]]) #true strain
stress = EStress[indx:maxStressIndex[0][0]]*(1+EStrain[indx:maxStressIndex[0][0]])# true sterss
TplStrain = strain -stress*(1/slope)
TplStress= stress
UTS_Engg = max(stress)
plt.plot((TplStrain),TplStress,'-r')
  # plt.set(xlim =[0,0.3],ylim = [0,500],title = 'TplStrain vs., TplStress')
plt.show()
  
###########################################
  # print("Please chose which order (1-9) equation you would like to fit to the TplStrain and TplStress curve")
fitting_func_order = input("Enter order of polynomial to fit - a number between 1-9 to continue...")
def get_fitting_function(func_order):
      match func_order:
          case "1":
              def objective(x, a, h, i):
                return a * x + h*np.sqrt(0.001+x) + i
              popt, _ = curve_fit(objective, TplStrain,TplStress)
              a, h, i = popt
              y_line = objective(TplStrain, a, h, i)
              return y_line            
          case "2":
              def objective(x, a, b, h, i):
                return a * x + b * x**2 + h*np.sqrt(0.001+x) + i
              popt, _ = curve_fit(objective, TplStrain,TplStress)
              a, b, h, i = popt
              y_line = objective(TplStrain, a, b, h, i)
              return y_line
          case "3":
              def objective(x, a, b, c, h, i):
                return a * x + b * x**2 + c*x**3+ h*np.sqrt(0.001+x) + i
              popt, _ = curve_fit(objective, TplStrain,TplStress)
              a, b, c, h, i = popt
              y_line = objective(TplStrain, a, b, c, h, i)
              return y_line
          case "4":
              def objective(x, a, b, c, d, h, i):
                return a * x + b * x**2 + c*x**3+ d*x**4+ h*np.sqrt(0.001+x) + i
              popt, _ = curve_fit(objective, TplStrain,TplStress)
              a, b, c, d, h, i = popt
              y_line = objective(TplStrain, a, b, c, d, h, i)
              return y_line
          case "5":
              def objective(x, a, b, c, d, e, h, i):
                return a * x + b * x**2 + c*x**3+ d*x**4+ e*x**5+ h*np.sqrt(0.001+x) + i
              popt, _ = curve_fit(objective, TplStrain,TplStress)
              a, b, c, d, e, h, i = popt
              y_line = objective(TplStrain, a, b, c, d, e, h, i)
              return y_line
          case "6":
              def objective(x, a, b, c, d, e, f, h, i):
                return a * x + b * x**2 + c*x**3+ d*x**4+ e*x**5+ f*x**6+ h*np.sqrt(0.001+x) + i
              popt, _ = curve_fit(objective, TplStrain,TplStress)
              a, b, c, d, e, f, h, i = popt
              y_line = objective(TplStrain, a, b, c, d, e, f, h, i)
              return y_line
          case "7":
              def objective(x, a, b, c, d, e, f, g, h, i):
                return a * x + b * x**2 + c*x**3 + d*x**4 + e*x**5 + f*x**6 + g*x**7 + h*np.sqrt(0.001+x) + i
              popt, _ = curve_fit(objective, TplStrain,TplStress)
              a, b, c, d, e, f, g, h, i = popt
              y_line = objective(TplStrain, a, b, c, d, e, f, g, h, i)
              return y_line
          case "8":
              def objective(x, a, b, c, d, e, f, g, h1, h, i):
                return a * x + b * x**2 + c*x**3 + d*x**4 + e*x**5 + f*x**6 + g*x**7 + h1*x**8+ h*np.sqrt(0.001+x) + i
              popt, _ = curve_fit(objective, TplStrain,TplStress)
              a, b, c, d, e, f, g, h1,h, i = popt
              y_line = objective(TplStrain, a, b, c, d, e, f, g, h1, h, i)
              return y_line
          case "9":
              def objective(x, a, b, c, d, e, f, g, h1, j1, h, i):
                return a * x + b * x**2 + c*x**3 + d*x**4 + e*x**5 + f*x**6 + g*x**7 + h1*x**8+ j1*x**9+ h*np.sqrt(0.001+x) + i
              popt, _ = curve_fit(objective, TplStrain,TplStress)
              a, b, c, d, e, f, g, h1, j1, h, i= popt
              y_line = objective(TplStrain, a, b, c, d, e, f, g, h1, j1, h, i)
              return y_line
          case _:
              raise ValueError(f"Invalid order of the equation")
plt.scatter(TplStrain, TplStress)
# plt.set(xlim=[0.0,0.3],ylim=[0,500])
y_line = get_fitting_function(fitting_func_order)
# y_line = objective(TplStrain, a, b, c, d, e, f, g, h, i)
plt.plot(TplStrain, y_line, color='red', marker='o', linestyle='dashed',linewidth=1, markersize=1)
plt.xlabel('True Stress')
plt.ylabel('True Plastic Strain')

plt.show()

KMY = np.gradient(y_line, TplStrain)
KMX = y_line-y_line[0]
plt.scatter(KMX,KMY)
plt.title ('d(sigma)/d(epsilon) vs sigma')
plt.xlabel( 'True Stress - Yield Stress (MPa)' )
plt.ylabel( 'd(sigma)/d(epsilon)') 
plt.show()
# for fitting line and getting intercepts on axes
input()
num1, num2 = input("Please enter the linear range separated by space: ").split()
lower = float(num1)
upper = float(num2)
id_lo = find_closest(KMX, lower)
id_up = find_closest(KMX, upper)
def objective_2(x, a, b):
  return a * x + b
popt1, _ = curve_fit(objective_2, KMX[id_lo:id_up],KMY[id_lo:id_up])
a, b = popt1
# figure1 = plt.plot()
plt.scatter(KMX,KMY,color='green')
plt.plot(KMX,objective_2(KMX, a, b),color='red')
plt.xlim([0,150])
print('y1 = %.5f * x + %.5f' % (a, b))
print('x-intercept1 = %.5f, y-intercept1 = %.5f'%(-b/a, b)) 
plt.show()
input()

  ###############
  ################

folderPath = "C:\\Users\\MKY\\OneDrive\\PythonWork\\"
csvFilename = 'Deepak_Data_B'+'_'+'.csv'
import csv

YS1= [YS]
UTS1= [UTS_Engg]
Modulus=[slope]
sigma_sat= [-b/a]
theta_0= [b]
fitorder= [fitting_func_order]
slope_theta0 = [a]
with open(csvFilename, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['YS','UTS','E','Sigma_sat','theta_0',
                     'slope_theta0','fitorder',
                     'KMX','KMY','TplStrain','TplStress','TruePlasticStrain'])
    for i in range(max(len(KMX), len(TplStrain))):
        writer.writerow([YS1[i] if i < len(YS1) else "",
                         UTS1[i] if i < len(UTS1) else "",
                         Modulus[i] if i < len(Modulus) else "",
                         sigma_sat[i] if i < len(sigma_sat) else "",
                         theta_0[i] if i < len(theta_0) else "",
                         slope_theta0[i] if i < len(slope_theta0) else "",
                         fitorder[i] if i < len(fitorder) else "",
                         KMX[i] if i < len(KMX) else "",
                         KMY[i] if i < len(KMY) else "",
                         TplStrain[i] if i < len(TplStrain) else "",
                         TplStress[i] if i < len(TplStress) else ""])