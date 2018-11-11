# Selective Inference after hierarchical clustering (ward)

## main.py は使用例

## method.py で定義されている主な関数の簡単な説明を示す

### ward()
- 階層型クラスタリングの1つであるウォード法を行う
- 引数：$n \times d$のデータ行列(ndarray型) 
  - $n$: サンプルサイズ
  - $d$: 特徴(次元)数
- 返り値: output, c_list_list, c_ab_list, a_list, b_list
  - output 
    - scipy.cluster.hierarchy.dendrogramに対応した$n - 1 \times 4$のndarray型配列を返し, この返り値をそのままdendrogramの引数として与えれば, 樹形図が表示される
    - 1つ目と2つ目の要素は, どのクラスタが統合されたのかを表す
    - 3つ目の要素は, 統合された2つのクラスタ間距離
    - 4つ目は統合されたクラスタの要素数
  - c_list_list, c_ab_list, a_list, b_list
    - pci-ward()の使用に必要な各ステップでのクラスタの情報 

### pci-ward()
- ward法後のSelective Infereceを行う
- 返り値: naive-p, selective-p
  - naive-p
    - クラスタリングの影響を考慮せずに検定を行った場合のp値を示す
  - selective-p 
    - Selective Infereceの枠組みを用いて計算したp値
  
### pv_dendrogram() `未完成版`
- scipy.cluster.hierarchy.dendrogramにp値を付与する
- これはまだ, 完全にはできておらず, データサイズが変わった場合に適宜位置を調節する必要がある