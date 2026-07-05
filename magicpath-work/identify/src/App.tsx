import { Theme } from './settings/types';
import { FruitIdentify } from './components/generated/FruitIdentify';

let theme: Theme = 'light';

function App() {
  function setTheme(theme: Theme) {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }

  setTheme(theme);

  return (
    <>
      <FruitIdentify />
    </>
  ); // %EXPORT_STATEMENT%
}

export default App;
