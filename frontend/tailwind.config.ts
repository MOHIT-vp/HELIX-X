import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './src/**/*.{ts,tsx}'],
  theme: { extend: { colors: { forest: '#283F24', leaf: '#467235', gold: '#FFBF00', cream: '#FFF78D', orange: '#E87F24' }, fontFamily: { display: ['Stardom', 'serif'], body: ['Satoshi', 'sans-serif'] } } },
  plugins: [],
}
export default config
