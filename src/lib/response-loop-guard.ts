export const LOOP_POLICY = "repeat-block-v1";
export function repeatedPassage(text: string) {
  const words = text.normalize("NFC").toLowerCase().match(/[\p{L}\p{N}]+(?:['\u2019][\p{L}\p{N}]+)?/gu) ?? [];
  for (let end = 32; end <= words.length; end++) {
    for (let width = 6; width <= Math.min(80, Math.floor(end / 2)); width++) {
      const copies = width < 24 ? 3 : 2;
      if (end < width * copies) continue;
      let same = true;
      for (let i = 1; i < copies && same; i++) {
        for (let j = 0; j < width; j++) {
          if (words[end - width + j] !== words[end - width * (i + 1) + j]) { same = false; break; }
        }
      }
      if (same) return { block_words: width, repetitions: copies, end_word: end };
    }
  }
  return null;
}
