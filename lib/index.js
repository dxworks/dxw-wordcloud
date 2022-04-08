const _package = require('../package.json')
const path = require('path')
const os = require('os')
const fs = require('fs')
const axios = require('axios')
const unzipper = require('unzipper')

const {execSync} = require('child_process')

const {Command} = require("commander")

DX_WC_CMD = "dx_wc"

exports.dxwcCommand = new Command()
  .name('wc')
  .description('Tool for generating word clouds based on the activity and commits messages of a codebase starting with the output of DX.')
  .option('--dx <dx_project_folder>', '[Optional] Points to the DX project directory, defaults to the current working directory.')
  .option('--max-words <max_words>', '[Optional] Maximum number of words to be added into the generated wordcloud, the words are ordered based on the amount of activity they describe, defaults to 20.')
  .option('--output <output_location>', '[Optional] Location where generate the output files, defatulst to `./+wordclouds/`')
  .allowUnknownOption()
  .action(runWordcloud)

async function runWordcloud(options) {
  const version = _package.version
  console.log('Checking if dx_wc==' + version + ' is installed')
  const currentVersionFolder = path.resolve(os.homedir(), '.dxw', DX_WC_CMD, _package.version)
  let platformName = getPlatformName();
  const cmdPath = path.resolve(currentVersionFolder, DX_WC_CMD + (platformName === 'win'? '.exe': ''));
  
  if (!fs.existsSync(cmdPath)) {
    console.log(`Command file ${cmdPath} does not exist, downloading...`)
    fs.mkdirSync(currentVersionFolder, {recursive: true})
    console.log(`Downloading ${DX_WC_CMD} ${_package.version}`)
    const downloadedFile = await downloadFile(
            `${_package.homepage}/download/v${_package.version}/${DX_WC_CMD}-${platformName}.zip`, 
            path.resolve(currentVersionFolder, `${DX_WC_CMD}.zip`))
    console.log(`Download Finished`)

    console.log('Installing...')
    await unzip(downloadedFile, {path: currentVersionFolder, overwriteRootDir: true})
    fs.chmodSync(cmdPath, '755')
    fs.rmSync(downloadedFile, {force: true})
    console.log('Install Finished')
  } else {
    console.log(`Found local installation at ${cmdPath}`)
  }

  myargs = ''
  Object.keys(options).forEach(function(key) {
    switch (key) {
        case 'maxWords':
            argk =  '--max-words'
            break;
        default:
            argk = '--' + key
    }
    myargs += ' ' + argk + ' ' + options[key]
  })

  await execSync(`${cmdPath} ${myargs}`, {cwd: process.cwd(), stdio: 'inherit'})
}

function getPlatformName() {
  switch (process.platform) {
    case 'win32':
      return 'win'
    case 'darwin':
      return 'osx'
    case 'linux':
      return 'linux'
    default:
      throw Error('Honeydew can only be installed on Windows, Mac or Linux systems')
  }
}

async function downloadFile(url, filename, payload, progressBar) {
  const file = fs.createWriteStream(filename, 'utf-8')
  let receivedBytes = 0

  const {data, headers, status} = await axios.get(url,
    {
      method: 'GET',
      responseType: 'stream',
    })

  const totalBytes = headers['content-length'] ? +headers['content-length'] : 0

  return new Promise((resolve, reject) => {
    if (status !== 200) {
      return reject('Response status was ' + status)
    }
    progressBar?.start(totalBytes, 0, payload)
    data
      .on('data', (chunk) => {
        receivedBytes += chunk.length
        progressBar?.update(receivedBytes, payload)
      })
      .pipe(file)
      .on('finish', () => {
        file.close()
        resolve(filename)
      })
      .on('error', (err) => {
        fs.unlinkSync(filename)
        progressBar?.stop()
        return reject(err)
      })
  })
}

async function unzip(zipFileName, options) {
  return new Promise((resolve, reject) => {
    if (options?.overwriteRootDir) {
      fs.createReadStream(zipFileName)
        .pipe(unzipper.Parse())
        .on('entry', function (entry) {
          const fullPathName = path.resolve(options.path, entry.path.substring(entry.path.indexOf('/') + 1, entry.path.length))
          if (entry.type === 'Directory') {
            if (!fs.existsSync(fullPathName))
              fs.mkdirSync(fullPathName, {recursive: true})
          } else
            entry.pipe(fs.createWriteStream(fullPathName))
        })
        .on('finish', () => {
          resolve()
        })
        .on('error', reject)
    } else {
      fs.createReadStream(zipFileName)
        .pipe(unzipper.Extract(options))
        .on('finish', () => {
          resolve()
        })
        .on('error', reject)
    }
  })
}