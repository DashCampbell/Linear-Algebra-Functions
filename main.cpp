#include <iostream>
#include <vector>

// matric structure
struct mat
{
public:
    float *m;

    mat(int rows, const int columns)
    {
        m = new float[rows][columns];
    }
    mat(float *matrix)
    {
        m = matrix;
    }
    ~mat()
    {
        delete m;
    }
    std::string toString() const
    {
        for (auto const &row : m)
        {
            std::cout << '[';
            for (auto const &col : row)
            {
                std::cout << m[row][col] << ", ";
            }
            std::cout << ']' << std::endl;
        }
    }
};

int main()
{
    mat M = new mat([ (float[])[1.0, 2.0], (float[])[3.0, 4.0] ]);
    std::cout << M.toString() << std::endl;

    return 0;
}