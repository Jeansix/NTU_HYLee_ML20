# <center> Tips for code

- loc vs iloc

  ```txt
  loc是根据dataframe的具体标签选取列，而iloc是根据标签的index（从0标记）选取列
  e.g.
  A    B    C    D
   
  0    ss   小红  8
  1    aa   小明  d
  4    f         f
  6    ak   小紫  7
  选取前三行的'A','C'标签：
  方法1.df = df.loc[0:2, ['A', 'C']]
  方法2.df = df.iloc[0:2, [0, 2]]
  ```

