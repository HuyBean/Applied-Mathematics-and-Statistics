class SystemsLinearEquations:
    def __init__(self):
        self.arr_input = []
        self.solutions = []
        self.numSolutions = 0
        self.numEquation = 0
        self.RowZero = None
    
    def getInut(self, filename :str):
        f = open(filename, 'r')
        self.numEquation = int(f.readline())

        for i in range(0, self.numEquation):
            self.arr_input.append([float(x) for x in f.readline().split()])
        f.close()

    def echelonMatrix(self):
        numLine = len(self.arr_input)
        for i in range(0, numLine):
            for j in range(i + 1, numLine):
                heso = -1
                isFullZero = False
                index = 0

                # Đổi dòng khác để tính toán
                while self.arr_input[i][i] == 0:
                    temp = self.arr_input[i]
                    self.arr_input.remove(temp)
                    self.arr_input.append(temp)  
                    index += 1
                    if index >= numLine:
                        isFullZero = True
                        break
                

                # index2 = i                            
                if self.arr_input[i][i] != 0:
                    heso = self.arr_input[j][i] / self.arr_input[i][i]

                if isFullZero and heso == -1:
                    continue
                isRowZero = True
                for k in range(len(self.arr_input[0])):
                    self.arr_input[j][k] = self.arr_input[j][k] - heso * self.arr_input[i][k]
                    
                    # self.arr_input[j][k] = round(self.arr_input[j][k], 3)
                    if self.arr_input[j][k] != 0:
                        isRowZero = False
                if isRowZero:
                    self.RowZero = self.arr_input[j]
        while self.RowZero in self.arr_input:
            self.arr_input.remove(self.RowZero)
        # for i in range(len(self.arr_input[0]) - 2 - numLine, numLine):
        #     for j in range(i + 1, numLine):
        #         for k in range(len(self.arr_input[0])):
        #             if self.arr_input[i][k] != 0 and self.arr_input[j][k] != 0:
        #                 heso = self.arr_input[j][k] / self.arr_input[i][k]
        #                 for l in range(len(self.arr_input[0])):
        #                     self.arr_input[j][l] = self.arr_input[j][l] - heso * self.arr_input[i][l]
        #                 break
        return
    def noSolution(self):
        for i in range(len(self.solutions)):
            self.solutions[i] = '?'
    def onlySolution(self, i, n, arr_solve, arr_input):

        if i < 0:
            return 
        elif i == n:
            if(arr_input[i][i] == 0):
                self.echelonMatrix()
            arr_solve[i] = arr_input[i][i + 1] / arr_input[i][i]
            self.onlySolution(i - 1, n, arr_solve, arr_input)
        else:
            arr_solve[i] = arr_input[i][n + 1]
            ii = n
            while i != ii:
                arr_solve[i] -= arr_input[i][ii] * arr_solve[ii]
                ii -= 1
            if(arr_input[i][i] == 0):
                self.echelonMatrix()
            if arr_input[i][i] == 0:
                self.noSolution()
                return
            arr_solve[i] /= arr_input[i][i]
            self.onlySolution(i - 1, n, arr_solve, arr_input)

    def manySolution(self, free, i, n, arr_solve, arr_input):
        if i < 0:
            return
        if free > 0:
            arr_solve[i] = 'a' + str(free)
            self.manySolution(free - 1, i - 1, n, arr_solve, arr_input)
        else:
            arr_solve[i] = str(arr_input[i][n + 1])
            ii = n
            while ii != i:
                
                arr_solve[i] = str(arr_solve[i]) + self.mulStr(arr_input[i][ii], arr_solve[ii])
                ii -= 1
            if arr_input[i][i] == 0:
                self.noSolution()
                return
            arr_solve[i] = '('+ self.devStr(arr_solve[i], arr_input[i][i]) + ')'
            self.manySolution(free, i - 1, n, arr_solve, arr_input)
    def Algorithm(self):
        for line in self.arr_input:
            print(line)
        print('-----------')
        self.echelonMatrix()
        for line in self.arr_input:
            print(line)
        self.numSolutions = len(self.arr_input[0]) - 1
        self.numEquation = len(self.arr_input)
        solutionFree = self.numSolutions - self.numEquation
        for i in range(self.numSolutions):
            self.solutions.append(None)

        if solutionFree == 0:
            self.onlySolution(self.numSolutions - 1, self.numSolutions - 1, self.solutions, self.arr_input)
            if self.solutions[0] == '?':
                print("No Solution")
            else:
                print("One Solution")

            return self.solutions
            
        elif solutionFree < 0:
            
            self.noSolution()
            return self.solutions
        else:
            self.manySolution(solutionFree, self.numSolutions - 1, self.numSolutions - 1, self.solutions, self.arr_input)
            if self.solutions[0] == '?':
                print("No Solution")
            else:
                print("Many Solutions")
            return self.solutions
    
    def clear(self):
        self.arr_input = []
        self.solutions = []
        self.numSolutions = 0
        self.numEquation = 0
        self.RowZero = None
    @staticmethod
    def subStr(number):
        if number >= 0:
            return (' - ' + str(number))
        return ' + ' + str(abs(number))
    @staticmethod
    def devStr(a, b):
        if b == 1:
            return str(a)
        elif b == -1:
            return '-(' + str(a) + ')'
        return str(a) + ' / ' + str(b)
    @staticmethod
    def mulStr(a, b):
        if a == 0:
            return ""
        elif a > 0:
            return ' - ' + str(a) + ' * ' + str(b)
        return ' + ' + str(-a) + ' * ' + str(b)

def main():
    sle = SystemsLinearEquations()

    filename = "Project1_test/input1.txt"
    sle.getInut(filename)
    result = sle.Algorithm()
    for i in range(len(result)):
        print('x[' + str(i + 1) + '] = ', result[i])
    sle.clear()

    print("\n|||||||||||||||||||||||||||||||\n")

    # filename = "input2.txt"
    # sle.getInut(filename)
    # result = sle.Algorithm()
    # for i in range(len(result)):
    #     print('x[' + str(i + 1) + '] = ', result[i])
    # sle.clear()

    # print("\n|||||||||||||||||||||||||||||||\n")

    # filename = "input5.txt"
    # sle.getInut(filename)
    # result = sle.Algorithm()
    # for i in range(len(result)):
    #     print('x[' + str(i + 1) + '] = ', result[i])
    # sle.clear()

    # print("\n|||||||||||||||||||||||||||||||\n")


main()
