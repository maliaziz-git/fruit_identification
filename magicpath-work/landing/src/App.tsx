import { Theme } from './settings/types';
import { SaaSLanding } from './components/generated/SaaSLanding';

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
      <SaaSLanding />
    </>
  ); // %EXPORT_STATEMENT%
}

export default App;
