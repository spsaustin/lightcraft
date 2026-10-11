//! The keyword vocabulary of the Describe lane: a few hundred original terms in facets (subjects, settings, genre,
//! light, colour, mood, weather, time of day, season), embedded as text prompts and scored against each photo's
//! SigLIP 2 embedding (`docs/vera-describe.md`, "Keywords by zero-shot").
//!
//! The terms live in `keywords.txt`, a small format of our own (no TOML crate needed):
//!
//! ```text
//! # comment lines start with #; blank lines are ignored
//! [subjects] multi max=8 prompt="This is a photo of {}."
//! mountains | mountain range
//! [time of day] single margin=0.02 prompt="This is a photo taken at {}."
//! dawn
//! ```
//!
//! A `[facet]` header names the facet, then `single` (at most one answer, taken only when it beats the runner-up by
//! `margin` in cosine) or `multi` (several, at most `max`), and an optional `prompt="…"` template holding `{}` for the
//! term. Each following line is one term; an optional ` | ` lists alternative phrasings that only build extra prompts
//! for the same keyword. The keyword written to a photo is the term itself.
//!
//! The parser treats its input as hostile (it is a file we ship today and may read from the config folder tomorrow):
//! size, line, term, facet and term-count caps, no control characters, and every problem is an `Err`, never a panic.
//! No term names a person's gender, ethnicity, nationality, age bracket, religion or body; the tests enforce it.

use std::collections::HashSet;

/// The whole file may not exceed this many bytes.
const MAX_FILE_BYTES: usize = 1 << 20;
/// A line may not exceed this many characters.
const MAX_LINE_CHARS: usize = 200;
/// A term, a phrasing or a facet name may not exceed this many characters.
const MAX_TERM_CHARS: usize = 60;
/// At most this many facets.
const MAX_FACETS: usize = 64;
/// At most this many terms in all.
const MAX_TERMS: usize = 2_000;
/// The largest `max=` a multi-answer facet may ask for.
const MAX_MULTI: usize = 64;
/// The template of a facet that names none.
const DEFAULT_PROMPT: &str = "This is a photo of {}.";
/// The margin of a single-answer facet that names none.
const DEFAULT_MARGIN: f32 = 0.02;
/// The cap of a multi-answer facet that names none.
const DEFAULT_MAX: usize = 8;
/// Separates a term from its alternative phrasings.
const ALTERNATIVES: &str = " | ";

/// How many answers a facet gives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Mode {
    /// At most one answer, and only when it beats the runner-up by `margin` in cosine.
    Single { margin: f32 },
    /// Several answers, at most `max`.
    Multi { max: usize },
}

/// One keyword and the phrasings that stand for it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Term {
    /// The keyword written to a photo.
    pub keyword: String,
    /// Every phrasing used to build prompts: the keyword first, then the alternatives.
    pub phrasings: Vec<String>,
}

/// A group of terms scored together with one prompt template and one answering rule.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Facet {
    pub name: String,
    pub mode: Mode,
    /// The prompt template; holds `{}` for the term.
    pub prompt: String,
    pub terms: Vec<Term>,
}

/// The parsed vocabulary.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Vocabulary {
    pub facets: Vec<Facet>,
}

impl Vocabulary {
    /// The vocabulary that ships with LightCraft. Never panics: a file that does not parse (a build mistake the tests
    /// catch) yields a vocabulary with no facets and a warning.
    pub fn builtin() -> Vocabulary {
        match Vocabulary::parse(include_str!("keywords.txt")) {
            Ok(vocabulary) => vocabulary,
            Err(e) => {
                log::warn!("vera: the built-in keyword vocabulary is unusable ({e}); Describe will propose no keywords");
                Vocabulary { facets: Vec::new() }
            }
        }
    }

    /// Parses the text strictly: a duplicate keyword anywhere in the file, an empty or repeated facet, a bad header, an
    /// unknown mode or option, a prompt without `{}`, a control character or an over-long line is an error naming the line.
    pub fn parse(text: &str) -> Result<Vocabulary, String> {
        if text.len() > MAX_FILE_BYTES {
            return Err(format!("the vocabulary is larger than {MAX_FILE_BYTES} bytes"));
        }
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let mut facets: Vec<Facet> = Vec::new();
        let mut keywords: HashSet<String> = HashSet::new();
        let mut total_terms = 0usize;
        for (index, raw) in text.lines().enumerate() {
            let n = index.saturating_add(1);
            if raw.chars().nth(MAX_LINE_CHARS).is_some() {
                return Err(line_error(n, &format!("the line is longer than {MAX_LINE_CHARS} characters")));
            }
            if raw.chars().any(char::is_control) {
                return Err(line_error(n, "the line holds a control character"));
            }
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with('[') {
                if facets.len() >= MAX_FACETS {
                    return Err(line_error(n, &format!("more than {MAX_FACETS} facets")));
                }
                if let Some(previous) = facets.last()
                    && previous.terms.is_empty()
                {
                    return Err(line_error(n, &format!("empty facet '{}'", previous.name)));
                }
                let facet = parse_header(line).map_err(|e| line_error(n, &e))?;
                if facets.iter().any(|f| f.name == facet.name) {
                    return Err(line_error(n, &format!("duplicate facet '{}'", facet.name)));
                }
                facets.push(facet);
            } else {
                let Some(facet) = facets.last_mut() else {
                    return Err(line_error(n, "a term before any facet header"));
                };
                if total_terms >= MAX_TERMS {
                    return Err(line_error(n, &format!("more than {MAX_TERMS} terms")));
                }
                let term = parse_term(line).map_err(|e| line_error(n, &e))?;
                if !keywords.insert(term.keyword.clone()) {
                    return Err(line_error(n, &format!("duplicate keyword '{}'", term.keyword)));
                }
                total_terms = total_terms.saturating_add(1);
                facet.terms.push(term);
            }
        }
        if let Some(last) = facets.last() {
            if last.terms.is_empty() {
                return Err(format!("empty facet '{}'", last.name));
            }
        } else {
            return Err("the vocabulary has no facets".to_string());
        }
        Ok(Vocabulary { facets })
    }

    /// How many terms there are in all facets.
    pub fn terms(&self) -> usize {
        self.facets.iter().map(|f| f.terms.len()).sum()
    }

    /// Every phrasing of every term rendered into its facet's template and lower-cased, as
    /// `(facet index, term index, prompt text)`.
    pub fn prompts(&self) -> Vec<(usize, usize, String)> {
        let mut out = Vec::new();
        for (facet_index, facet) in self.facets.iter().enumerate() {
            for (term_index, term) in facet.terms.iter().enumerate() {
                for phrasing in &term.phrasings {
                    out.push((facet_index, term_index, facet.prompt.replace("{}", phrasing).to_lowercase()));
                }
            }
        }
        out
    }

    /// A stable 64-bit FNV-1a hash of the normalised content (names, modes, templates, terms and phrasings, in order;
    /// not comments or spacing), so a cache of embedded prompts can be keyed by it.
    pub fn hash(&self) -> u64 {
        let mut h = Fnv::new();
        h.number(self.facets.len());
        for facet in &self.facets {
            h.text(&facet.name);
            match facet.mode {
                Mode::Single { margin } => {
                    h.bytes(&[1]);
                    h.bytes(&margin.to_bits().to_le_bytes());
                }
                Mode::Multi { max } => {
                    h.bytes(&[2]);
                    h.number(max);
                }
            }
            h.text(&facet.prompt);
            h.number(facet.terms.len());
            for term in &facet.terms {
                h.number(term.phrasings.len());
                for phrasing in &term.phrasings {
                    h.text(phrasing);
                }
            }
        }
        h.0
    }
}

/// FNV-1a, 64 bit.
struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn number(&mut self, n: usize) {
        self.bytes(&u64::try_from(n).unwrap_or(u64::MAX).to_le_bytes());
    }

    /// Length-prefixed, so `["ab", "c"]` and `["a", "bc"]` hash differently.
    fn text(&mut self, s: &str) {
        self.number(s.len());
        self.bytes(s.as_bytes());
    }
}

fn line_error(n: usize, message: &str) -> String {
    format!("line {n}: {message}")
}

/// Collapses runs of whitespace to one space and trims.
fn normalise(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A character a term or phrasing may hold: a lower-case letter or digit, a space, a hyphen, an apostrophe.
fn is_term_char(c: char) -> bool {
    (c.is_alphanumeric() && !c.is_uppercase()) || matches!(c, ' ' | '-' | '\'')
}

fn parse_header(line: &str) -> Result<Facet, String> {
    let rest = line.strip_prefix('[').ok_or("a facet header starts with '['")?;
    let (name, rest) = rest.split_once(']').ok_or("a facet header needs a closing ']'")?;
    let name = normalise(name);
    let name_ok = !name.is_empty()
        && name.chars().count() <= MAX_TERM_CHARS
        && name.chars().all(|c| (c.is_alphabetic() && !c.is_uppercase()) || matches!(c, ' ' | '-'));
    if !name_ok {
        return Err(format!("bad facet name '{name}' (lower-case words, at most {MAX_TERM_CHARS} characters)"));
    }
    let (options, prompt) = match rest.split_once("prompt=") {
        Some((before, after)) => {
            let after = after.strip_prefix('"').ok_or("prompt= needs a quoted template")?;
            let (value, tail) = after.split_once('"').ok_or("unterminated prompt template")?;
            (format!("{before} {tail}"), value.to_string())
        }
        None => (rest.to_string(), DEFAULT_PROMPT.to_string()),
    };
    if !prompt.contains("{}") {
        return Err("the prompt template needs '{}' for the term".to_string());
    }
    let mut words = options.split_whitespace();
    let mode_word = words.next().ok_or("missing mode (single or multi)")?;
    let mut margin: Option<f32> = None;
    let mut max: Option<usize> = None;
    for word in words {
        if let Some(value) = word.strip_prefix("margin=") {
            let m: f32 = value.parse().map_err(|_| format!("bad margin '{value}'"))?;
            if !m.is_finite() || !(0.0..=1.0).contains(&m) {
                return Err(format!("margin '{value}' is not between 0 and 1"));
            }
            margin = Some(m);
        } else if let Some(value) = word.strip_prefix("max=") {
            let m: usize = value.parse().map_err(|_| format!("bad max '{value}'"))?;
            if !(1..=MAX_MULTI).contains(&m) {
                return Err(format!("max '{value}' is not between 1 and {MAX_MULTI}"));
            }
            max = Some(m);
        } else {
            return Err(format!("unknown option '{word}'"));
        }
    }
    let mode = match mode_word {
        "single" => {
            if max.is_some() {
                return Err("a single-answer facet takes margin=, not max=".to_string());
            }
            Mode::Single { margin: margin.unwrap_or(DEFAULT_MARGIN) }
        }
        "multi" => {
            if margin.is_some() {
                return Err("a multi-answer facet takes max=, not margin=".to_string());
            }
            Mode::Multi { max: max.unwrap_or(DEFAULT_MAX) }
        }
        other => return Err(format!("unknown mode '{other}' (single or multi)")),
    };
    Ok(Facet { name, mode, prompt, terms: Vec::new() })
}

fn parse_term(line: &str) -> Result<Term, String> {
    let mut phrasings: Vec<String> = Vec::new();
    for part in line.split(ALTERNATIVES) {
        let phrase = normalise(part);
        if phrase.is_empty() || !phrase.chars().any(char::is_alphanumeric) {
            return Err("an empty term or phrasing".to_string());
        }
        if phrase.chars().nth(MAX_TERM_CHARS).is_some() {
            return Err(format!("'{phrase}' is longer than {MAX_TERM_CHARS} characters"));
        }
        if !phrase.chars().all(is_term_char) {
            return Err(format!("'{phrase}' must be lower-case letters, digits, spaces, hyphens or apostrophes"));
        }
        if !phrasings.contains(&phrase) {
            phrasings.push(phrase);
        }
    }
    let keyword = phrasings.first().cloned().ok_or("an empty term")?;
    Ok(Term { keyword, phrasings })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = "[things] multi\nrock\nriver | a river\n";

    fn facet<'a>(v: &'a Vocabulary, name: &str) -> &'a Facet {
        v.facets.iter().find(|f| f.name == name).unwrap_or_else(|| panic!("no facet {name}"))
    }

    #[test]
    fn builtin_parses_with_the_planned_size() {
        let v = Vocabulary::builtin();
        assert!((8..=12).contains(&v.facets.len()), "{} facets", v.facets.len());
        assert!((380..=450).contains(&v.terms()), "{} terms", v.terms());
        assert_eq!(v.terms(), v.facets.iter().map(|f| f.terms.len()).sum::<usize>());
        for f in &v.facets {
            assert!(!f.terms.is_empty(), "facet {} is empty", f.name);
            assert!(f.prompt.contains("{}"), "facet {} prompt has no {{}}", f.name);
            for t in &f.terms {
                assert_eq!(t.phrasings.first(), Some(&t.keyword), "the keyword leads its phrasings");
                assert!(t.keyword.chars().count() <= MAX_TERM_CHARS);
                assert!(t.keyword.chars().all(is_term_char), "{}", t.keyword);
            }
        }
        // The planned shape: subjects are the biggest facet, time of day has six terms and season four.
        assert!(facet(&v, "subjects").terms.len() >= 150);
        assert_eq!(facet(&v, "time of day").terms.len(), 6);
        assert_eq!(facet(&v, "season").terms.len(), 4);
    }

    #[test]
    fn builtin_facet_modes() {
        let v = Vocabulary::builtin();
        for name in ["time of day", "season"] {
            match facet(&v, name).mode {
                Mode::Single { margin } => assert!((margin - 0.02).abs() < 1e-6, "{name} margin {margin}"),
                other => panic!("{name} should be single-answer, not {other:?}"),
            }
        }
        assert_eq!(facet(&v, "colour").mode, Mode::Multi { max: 3 });
        assert_eq!(facet(&v, "subjects").mode, Mode::Multi { max: 8 });
        for f in &v.facets {
            if let Mode::Multi { max } = f.mode {
                assert!((1..=8).contains(&max), "facet {} max {max}", f.name);
            }
        }
    }

    #[test]
    fn builtin_has_no_duplicate_keywords() {
        let v = Vocabulary::builtin();
        let mut seen = HashSet::new();
        for f in &v.facets {
            for t in &f.terms {
                assert!(seen.insert(t.keyword.clone()), "duplicate keyword {}", t.keyword);
            }
        }
        assert_eq!(seen.len(), v.terms());
    }

    #[test]
    fn builtin_contains_every_demo_scene_keyword() {
        // `lightcraft-scenes` is a regular dependency of the engine, so the real library is used, not a copy.
        let v = Vocabulary::builtin();
        let all: HashSet<&str> = v.facets.iter().flat_map(|f| f.terms.iter().map(|t| t.keyword.as_str())).collect();
        let scenes = lightcraft_scenes::demo_library();
        assert!(!scenes.is_empty());
        let mut checked = 0;
        for scene in &scenes {
            for k in &scene.meta.keywords {
                assert!(all.contains(k), "demo scene '{}' keyword '{k}' is not in the vocabulary", scene.name);
                checked += 1;
            }
        }
        assert!(checked >= 24, "only {checked} scene keywords were checked");
    }

    #[rustfmt::skip]
    const BANNED_WORDS: &[&str] = &[
        "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "lady", "ladies", "guy", "guys", "male", "female", "gentleman",
        "asian", "african", "european", "american", "indian", "chinese", "japanese", "latino", "hispanic", "arab", "caucasian",
        "elderly", "senior", "seniors", "teen", "teens", "teenager", "toddler", "baby", "infant", "young", "adult", "adults",
        "disabled", "disability", "wheelchair", "blind", "deaf", "muslim", "christian", "jewish", "hindu", "buddhist", "atheist",
        "fat", "thin", "obese", "skinny", "slim", "sexy", "sex", "nude", "naked", "erotic", "porn", "gay", "lesbian", "trans",
    ];
    const BANNED_PHRASES: &[&str] = &["black man", "white woman", "old man", "old woman", "black people", "white people"];

    /// Why `text`, a keyword or phrasing of `facet`, is not allowed (`None` when it is).
    fn banned_hit(text: &str, facet: &str) -> Option<String> {
        for phrase in BANNED_PHRASES {
            if text.contains(phrase) {
                return Some(format!("phrase '{phrase}'"));
            }
        }
        for token in text.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()) {
            if BANNED_WORDS.contains(&token) {
                return Some(format!("word '{token}'"));
            }
            // "black" and "white" are colours: allowed in the colour facet and in the technique "black and white".
            if (token == "black" || token == "white") && facet != "colour" && !text.contains("black and white") {
                return Some(format!("colour word '{token}' outside the colour facet"));
            }
        }
        None
    }

    #[test]
    fn banned_check_catches_what_it_should() {
        assert!(banned_hit("woman", "subjects").is_some());
        assert!(banned_hit("a young child", "subjects").is_some());
        assert!(banned_hit("black man", "colour").is_some());
        assert!(banned_hit("white", "subjects").is_some());
        assert!(banned_hit("white", "colour").is_none());
        assert!(banned_hit("black and white", "genre and technique").is_none());
        assert!(banned_hit("a man-made lake", "subjects").is_some());
        assert!(banned_hit("mountains", "subjects").is_none());
    }

    #[test]
    fn builtin_has_no_protected_attribute_or_sexual_terms() {
        let v = Vocabulary::builtin();
        for f in &v.facets {
            for t in &f.terms {
                for phrasing in &t.phrasings {
                    if let Some(why) = banned_hit(phrasing, &f.name) {
                        panic!("facet {}, term '{}', phrasing '{phrasing}': {why}", f.name, t.keyword);
                    }
                }
            }
            if let Some(why) = banned_hit(&f.prompt.to_lowercase(), &f.name) {
                panic!("facet {} prompt: {why}", f.name);
            }
        }
    }

    #[test]
    fn prompts_render_the_template_lower_cased() {
        // The template keeps its capitals in the file; the rendered prompt is lower-cased.
        let text = "[time of day] single margin=0.05 prompt=\"This is a Photo taken at {}.\"\ndawn | first light\n[things] multi\nrock\n";
        let v = Vocabulary::parse(text).unwrap();
        assert_eq!(v.facets[0].prompt, "This is a Photo taken at {}.");
        let prompts = v.prompts();
        assert_eq!(
            prompts,
            vec![
                (0, 0, "this is a photo taken at dawn.".to_string()),
                (0, 0, "this is a photo taken at first light.".to_string()),
                (1, 0, "this is a photo of rock.".to_string()),
            ]
        );
    }

    #[test]
    fn builtin_prompts_are_complete() {
        let v = Vocabulary::builtin();
        let prompts = v.prompts();
        let phrasings: usize = v.facets.iter().flat_map(|f| &f.terms).map(|t| t.phrasings.len()).sum();
        assert_eq!(prompts.len(), phrasings);
        assert!(prompts.len() >= v.terms());
        for (fi, ti, text) in &prompts {
            assert!(!text.contains("{}"), "unrendered prompt '{text}'");
            assert_eq!(text, &text.to_lowercase());
            assert!(v.facets.get(*fi).and_then(|f| f.terms.get(*ti)).is_some());
        }
        assert!(prompts.iter().any(|(_, _, t)| t == "this is a photo of mountains."));
        assert!(prompts.iter().any(|(_, _, t)| t == "this is a photo taken at dusk."));
    }

    #[test]
    fn hash_is_stable_and_follows_content() {
        let text = include_str!("keywords.txt");
        let a = Vocabulary::parse(text).unwrap();
        let b = Vocabulary::parse(text).unwrap();
        assert_eq!(a.hash(), b.hash());
        assert_eq!(a.hash(), Vocabulary::builtin().hash());
        assert_ne!(a.hash(), 0);

        // Comments, blank lines, CRLF and spacing do not change it.
        let noisy = format!("# a new comment\r\n\r\n{}", text.replace('\n', "\r\n"));
        assert_eq!(Vocabulary::parse(&noisy).unwrap().hash(), a.hash());

        // A changed term, an added phrasing, a changed template, mode or margin all do.
        let changed = text.replace("\nmountains |", "\nmountainz |");
        assert_ne!(changed, text);
        assert_ne!(Vocabulary::parse(&changed).unwrap().hash(), a.hash());
        let more = text.replacen("\nwaterfall | a waterfall", "\nwaterfall | a waterfall | a cascade", 1);
        assert_ne!(more, text);
        assert_ne!(Vocabulary::parse(&more).unwrap().hash(), a.hash());
        let prompt = text.replacen(
            "[season] single margin=0.02 prompt=\"This is a photo taken in {}.\"",
            "[season] single margin=0.02 prompt=\"A photo from {}.\"",
            1,
        );
        assert_ne!(prompt, text);
        assert_ne!(Vocabulary::parse(&prompt).unwrap().hash(), a.hash());
        let margin = text.replacen("[season] single margin=0.02", "[season] single margin=0.03", 1);
        assert_ne!(margin, text);
        assert_ne!(Vocabulary::parse(&margin).unwrap().hash(), a.hash());

        // Length-prefixing: moving a boundary between two terms changes the hash.
        let x = Vocabulary::parse("[f] multi\nab\nc\n").unwrap().hash();
        let y = Vocabulary::parse("[f] multi\na\nbc\n").unwrap().hash();
        assert_ne!(x, y);
    }

    #[test]
    fn parses_a_minimal_file_with_defaults() {
        let v = Vocabulary::parse(MINIMAL).unwrap();
        assert_eq!(v.facets.len(), 1);
        let f = &v.facets[0];
        assert_eq!(f.name, "things");
        assert_eq!(f.mode, Mode::Multi { max: DEFAULT_MAX });
        assert_eq!(f.prompt, DEFAULT_PROMPT);
        assert_eq!(f.terms[0], Term { keyword: "rock".into(), phrasings: vec!["rock".into()] });
        assert_eq!(f.terms[1], Term { keyword: "river".into(), phrasings: vec!["river".into(), "a river".into()] });
        assert_eq!(v.terms(), 2);

        let s = Vocabulary::parse("[x] single\nrock\n").unwrap();
        assert_eq!(s.facets[0].mode, Mode::Single { margin: DEFAULT_MARGIN });
    }

    #[test]
    fn parser_is_forgiving_about_layout_only() {
        let v =
            Vocabulary::parse("\u{feff}# comment\r\n\r\n  [ big   things ] multi max=3 \r\n  rock   face |  rock  face | a  cliff  \r\n").unwrap();
        assert_eq!(v.facets[0].name, "big things");
        assert_eq!(v.facets[0].mode, Mode::Multi { max: 3 });
        // Whitespace is collapsed and a repeated phrasing is kept once.
        assert_eq!(v.facets[0].terms[0].phrasings, vec!["rock face".to_string(), "a cliff".to_string()]);
    }

    #[test]
    fn strict_parser_rejects_each_malformed_input() {
        let long_line = format!("# {}", "x".repeat(MAX_LINE_CHARS));
        let long_term = format!("[f] multi\n{}\n", "x".repeat(MAX_TERM_CHARS + 1));
        let long_phrasing = format!("[f] multi\nrock | {}\n", "y".repeat(MAX_TERM_CHARS + 1));
        let cases: Vec<(String, &str)> = vec![
            ("".into(), "no facets"),
            ("# only a comment\n\n".into(), "no facets"),
            ("rock\n".into(), "before any facet"),
            ("[f multi\nrock\n".into(), "closing"),
            ("[] multi\nrock\n".into(), "bad facet name"),
            ("[Upper] multi\nrock\n".into(), "bad facet name"),
            ("[a1] multi\nrock\n".into(), "bad facet name"),
            ("[f]\nrock\n".into(), "missing mode"),
            ("[f] several\nrock\n".into(), "unknown mode"),
            ("[f] sing\nrock\n".into(), "unknown mode"),
            ("[f] multi colour=3\nrock\n".into(), "unknown option"),
            ("[f] multi max=0\nrock\n".into(), "max"),
            ("[f] multi max=65\nrock\n".into(), "max"),
            ("[f] multi max=-1\nrock\n".into(), "bad max"),
            ("[f] multi max=abc\nrock\n".into(), "bad max"),
            ("[f] multi margin=0.1\nrock\n".into(), "max="),
            ("[f] single max=3\nrock\n".into(), "margin="),
            ("[f] single margin=nan\nrock\n".into(), "margin"),
            ("[f] single margin=inf\nrock\n".into(), "margin"),
            ("[f] single margin=-0.1\nrock\n".into(), "margin"),
            ("[f] single margin=2\nrock\n".into(), "margin"),
            ("[f] single margin=x\nrock\n".into(), "bad margin"),
            ("[f] multi prompt=\"no braces here\"\nrock\n".into(), "'{}'"),
            ("[f] multi prompt=\"\"\nrock\n".into(), "'{}'"),
            ("[f] multi prompt=\"unterminated {}\nrock\n".into(), "unterminated"),
            ("[f] multi prompt=bare {}\nrock\n".into(), "quoted"),
            ("[f] multi prompt=\"a {}\" prompt=\"b {}\"\nrock\n".into(), "unknown option"),
            ("[f] multi\n".into(), "empty facet"),
            ("[f] multi\nrock\n[g] multi\n".into(), "empty facet"),
            ("[f] multi\n[g] multi\nrock\n".into(), "empty facet"),
            ("[f] multi\nrock\n[g] multi\nrock\n".into(), "duplicate keyword"),
            ("[f] multi\nrock\nrock\n".into(), "duplicate keyword"),
            ("[f] multi\nrock\n[f] multi\nriver\n".into(), "duplicate facet"),
            ("[f] multi\nrock\u{7}\n".into(), "control character"),
            ("[f] multi\nrock\tface\n".into(), "control character"),
            ("[f] multi\nrock\rface\n".into(), "control character"),
            ("[f] multi\nrock\u{0}\n".into(), "control character"),
            ("# fine\u{1b}[31m\n[f] multi\nrock\n".into(), "control character"),
            ("[f] multi\nRock\n".into(), "lower-case"),
            ("[f] multi\nro{}ck\n".into(), "lower-case"),
            ("[f] multi\nrock |\n".into(), "lower-case"),
            ("[f] multi\nrock |  | cliff\n".into(), "empty"),
            ("[f] multi\n| rock\n".into(), "lower-case"),
            ("[f] multi\n---\n".into(), "empty"),
            ("[f] multi\nrock | Cliff\n".into(), "lower-case"),
            (long_line.clone() + "\n[f] multi\nrock\n", "longer than 200"),
            (format!("[f] multi\n{}\n", "x".repeat(MAX_LINE_CHARS + 1)), "longer than 200"),
            (format!("[f] multi prompt=\"{} {{}}\"\nrock\n", "w".repeat(MAX_LINE_CHARS)), "longer than 200"),
            (long_term, "longer than 60"),
            (long_phrasing, "longer than 60"),
            ("#".repeat(MAX_FILE_BYTES + 1), "larger than"),
        ];
        for (input, fragment) in &cases {
            match Vocabulary::parse(input) {
                Ok(_) => panic!("accepted {input:?}"),
                Err(e) => assert!(e.contains(fragment), "input {input:?}: error {e:?} lacks {fragment:?}"),
            }
        }
        // Exactly at the limits is fine.
        let at_limit = format!("[f] multi\n{}\n", "x".repeat(MAX_TERM_CHARS));
        assert!(Vocabulary::parse(&at_limit).is_ok());
        let comment_at_limit = format!("# {}\n[f] multi\nrock\n", "x".repeat(MAX_LINE_CHARS - 2));
        assert!(Vocabulary::parse(&comment_at_limit).is_ok());
    }

    #[test]
    fn strict_parser_enforces_the_count_caps() {
        let build = |n: usize| (0..n).map(|i| format!("[facet {}] multi\nterm {}\n", letters(i), letters(i))).collect::<String>();
        assert!(Vocabulary::parse(&build(MAX_FACETS)).is_ok());
        let e = Vocabulary::parse(&build(MAX_FACETS + 1)).unwrap_err();
        assert!(e.contains("more than 64 facets"), "{e}");

        let terms = |n: usize| {
            let mut s = String::from("[big] multi\n");
            for i in 0..n {
                s.push_str(&format!("term {}\n", letters(i)));
            }
            s
        };
        assert_eq!(Vocabulary::parse(&terms(MAX_TERMS)).unwrap().terms(), MAX_TERMS);
        let e = Vocabulary::parse(&terms(MAX_TERMS + 1)).unwrap_err();
        assert!(e.contains("more than 2000 terms"), "{e}");
    }

    /// `0 -> "a"`, `1 -> "b"`, `26 -> "ba"`: lower-case letters only, so names and terms stay valid.
    fn letters(mut n: usize) -> String {
        let mut out = Vec::new();
        loop {
            out.push(char::from(b'a' + u8::try_from(n % 26).unwrap_or(0)));
            n /= 26;
            if n == 0 {
                break;
            }
        }
        out.iter().rev().collect()
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            usize::try_from(self.next() % u64::try_from(n.max(1)).unwrap_or(1)).unwrap_or(0)
        }
    }

    #[test]
    fn fuzz_never_panics() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        // 2,000 random byte strings, made valid UTF-8 by lossy decoding (the parser takes `&str`).
        for _ in 0..2_000 {
            let len = rng.below(400);
            let bytes: Vec<u8> = (0..len).map(|_| (rng.next() >> 24) as u8).collect();
            let _ = Vocabulary::parse(&String::from_utf8_lossy(&bytes));
        }
        // Token soup, which reaches the header and term code far more often than random bytes do.
        let long = "x".repeat(70);
        let pieces = [
            "[",
            "]",
            "[a]",
            "[big things]",
            " single",
            " multi",
            " max=",
            " margin=",
            " prompt=\"",
            "\"",
            "{}",
            "3",
            "0.02",
            "nan",
            "-1",
            "\n",
            "\r\n",
            " | ",
            "#",
            "rock",
            "river",
            "a",
            " ",
            "  ",
            "\t",
            "\u{0}",
            "é",
            "日本",
            "\u{feff}",
            long.as_str(),
        ];
        for _ in 0..2_000 {
            let mut s = String::new();
            for _ in 0..rng.below(40) {
                s.push_str(pieces.get(rng.below(pieces.len())).copied().unwrap_or(""));
            }
            let _ = Vocabulary::parse(&s);
        }
        // Single-byte corruptions and truncations of the real file.
        let text = include_str!("keywords.txt");
        let bytes = text.as_bytes();
        for _ in 0..300 {
            let mut copy = bytes.to_vec();
            for _ in 0..=rng.below(4) {
                let at = rng.below(copy.len());
                if let Some(b) = copy.get_mut(at) {
                    *b = (rng.next() >> 16) as u8;
                }
            }
            let cut = rng.below(copy.len() + 1);
            copy.truncate(cut);
            let _ = Vocabulary::parse(&String::from_utf8_lossy(&copy));
        }
        // And every prefix at line granularity: a half-written file is an `Err` or a shorter vocabulary, not a panic.
        for (i, _) in text.match_indices('\n').step_by(7) {
            let _ = Vocabulary::parse(text.get(..i).unwrap_or(""));
        }
    }
}
