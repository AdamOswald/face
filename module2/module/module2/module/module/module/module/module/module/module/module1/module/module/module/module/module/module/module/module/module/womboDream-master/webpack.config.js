const path = require('path');

module.exports = {
    entry: './src/index.js',
    
    output: {
        filename: 'bundle.js',
        path: path.resolve(__dirname, 'dist')
    },
    module: {
        rules: [
          {
            test: /\.?js/,
            exclude: /node_modules/,
            use: {
              loader: "babel-loader",
              options: {
                presets: ['@babel/preset-env', '@babel/preset-react']
              }
            }
          },
        ],
      },
      resolve: {
            extensions: [".js", ".jsx"],
            fallback: {
              "fs": false
          },
    },
    devServer: {
        port: 1200,
        static: path.resolve(__dirname, 'dist'),
        liveReload: true,
    },
    mode: "development"
}