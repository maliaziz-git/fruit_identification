import { Theme } from './settings/types';
import { FruitEncyclopediaGrid } from './components/generated/FruitEncyclopediaGrid';

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
      <FruitEncyclopediaGrid />
    </>
  ); // %EXPORT_STATEMENT%
}

export default App;
